using UnityEngine;
using Unity.InferenceEngine;
using FF = Unity.InferenceEngine.Functional;
using System.IO;
using System.Collections.Generic;
using System.Diagnostics;
using Debug = UnityEngine.Debug;
using System.Threading.Tasks;
using System;

public class ModelVLM : MonoBehaviour
{
    [Header("Model Settings")]
    public BackendType BACKEND = BackendType.GPUCompute;
    private const int MAX_GENERATE_TOKENS = 256;

    [Header("Model Assets")]
    [SerializeField] private ModelAsset visionEncoderAsset;
    [SerializeField] private ModelAsset embedTokensAsset;
    [SerializeField] private ModelAsset decoderAsset;

    private Worker _visionEncoder;
    private Worker _embedTokens;
    private Worker _decoder;
    private Worker _greedyDecoder;
    private Worker _concatEmbeddings;
    private Qwen2Tokenizer _tokenizer;
    private int _concatEmbeddingDimension = -1;

    private const int MAX_LAYERS = 24;
    private const int NUM_KEY_VALUE_HEADS = 2;
    private const int HEAD_DIM = 64;
    private const int VOCAB_SIZE = 151646;

    private Tensor<float>[] _pastKeys = new Tensor<float>[MAX_LAYERS];
    private Tensor<float>[] _pastValues = new Tensor<float>[MAX_LAYERS];
    private List<int> _outputTokens = new List<int>();

    public bool IsInitialized { get; private set; }
    public bool IsGenerating { get; private set; }

    public event Action<string> OnTokenGenerated;
    public event Action<string, int, long> OnGenerationComplete;
    public event Action<string> OnGenerationError;

    public async Task Initialize()
    {
        IsInitialized = false;
        try
        {
            Debug.Log("[ModelVLM] Starting initialization...");

            if (visionEncoderAsset == null || embedTokensAsset == null || decoderAsset == null)
            {
                Debug.LogError("[ModelVLM] Missing ModelAssets in Inspector!");
                return;
            }

            DisposeWorkers();

            // --- ANDROID COMPATIBLE TOKENIZER LOADING ---
            string tokenizerFolder = "fastvlm";
        
            Debug.Log("[ModelVLM] Loading Tokenizer Files from StreamingAssets...");
            string vocabJson = await GetStreamingAssetsText(Path.Combine(tokenizerFolder, "vocab.json"));
            string mergesTxt = await GetStreamingAssetsText(Path.Combine(tokenizerFolder, "merges.txt"));
            string configJson = await GetStreamingAssetsText(Path.Combine(tokenizerFolder, "tokenizer_config.json"));

            if (string.IsNullOrEmpty(vocabJson) || string.IsNullOrEmpty(mergesTxt))
            {
                Debug.LogError("[ModelVLM] Failed to load tokenizer files. Check StreamingAssets folder!");
                return;
            }

            _tokenizer = new Qwen2Tokenizer(vocabJson, mergesTxt, configJson);
            Debug.Log($"[ModelVLM] Tokenizer loaded. EOS: {_tokenizer.EosTokenId}");

            // --- MODEL LOADING (Sentis handles paths automatically) ---
            _visionEncoder = new Worker(ModelLoader.Load(visionEncoderAsset), BACKEND);
            _embedTokens = new Worker(ModelLoader.Load(embedTokensAsset), BACKEND);
            _decoder = new Worker(ModelLoader.Load(decoderAsset), BACKEND);

            // Create greedy decoder graph
            FunctionalGraph graph = new FunctionalGraph();
            FunctionalTensor logitsInput = graph.AddInput<float>(new DynamicTensorShape(1, -1, VOCAB_SIZE));
            FunctionalTensor argMax = FF.ArgMax(logitsInput, 2, false);
            _greedyDecoder = new Worker(graph.Compile(argMax), BACKEND);

            IsInitialized = true;
            Debug.Log("[ModelVLM] ✓✓✓ All models initialized successfully on Quest 3! ✓✓✓");
        }
        catch (Exception e)
        {
            Debug.LogError($"[ModelVLM] INIT FAILED: {e.Message}\n{e.StackTrace}");
            DisposeWorkers();
        }
    }

    private async Task<string> GetStreamingAssetsText(string fileName)
    {
        string path = Path.Combine(Application.streamingAssetsPath, fileName);

        // On Android, we must use WebRequest
        if (path.Contains("://") || path.Contains("jar:file"))
        {
            using (UnityEngine.Networking.UnityWebRequest www = UnityEngine.Networking.UnityWebRequest.Get(path))
            {
                var operation = www.SendWebRequest();
                while (!operation.isDone) await Task.Yield();

                if (www.result != UnityEngine.Networking.UnityWebRequest.Result.Success)
                {
                    Debug.LogError($"[ModelVLM] Error loading {fileName}: {www.error}");
                    return null;
                }
                return www.downloadHandler.text;
            }
        }
        else // Standard PC/Editor path
        {
            return File.ReadAllText(path);
        }
    }

    public async Task GenerateFromPrompt(string prompt, Texture image = null, int maxTokens = MAX_GENERATE_TOKENS)
    {
        if (string.IsNullOrEmpty(prompt)) return;
        if (!IsInitialized)
        {
            OnGenerationError?.Invoke("Models not initialized.");
            return;
        }
        if (IsGenerating)
        {
            OnGenerationError?.Invoke("Generation already in progress.");
            return;
        }

        maxTokens = Math.Max(1, maxTokens);

        IsGenerating = true;
        _outputTokens.Clear();
        ClearKVCache();

        Stopwatch sw = Stopwatch.StartNew();
        Tensor<float> visionEmbeddings = null;
        Tensor<float> mergedEmbeddings = null;

        try
        {
            mergedEmbeddings = BuildPromptEmbeddings(prompt, image, out visionEmbeddings);

            int mergedSeqLen = mergedEmbeddings.shape[1];
            int maxKvSequenceLength = mergedSeqLen + maxTokens;
            int nextToken = DecoderPrefill(mergedEmbeddings, mergedSeqLen, maxKvSequenceLength);

            int currentPos = mergedSeqLen;
            int generatedCount = 0;
            do
            {
                _outputTokens.Add(nextToken);
                generatedCount++;

                string decodedText = _tokenizer.Decode(_outputTokens);
                OnTokenGenerated?.Invoke(decodedText);

                nextToken = DecoderDecode(nextToken, currentPos);
                UpdateKVCache();
                currentPos++;
                await Task.Yield();
            }
            while (nextToken != _tokenizer.EosTokenId && nextToken != _tokenizer.PadTokenId && _outputTokens.Count < maxTokens);

            sw.Stop();
            string finalText = _tokenizer.Decode(_outputTokens);
            OnGenerationComplete?.Invoke(finalText, generatedCount, sw.ElapsedMilliseconds);
        }
        catch (Exception e)
        {
            Debug.LogError($"Generation error: {e.Message}\n{e.StackTrace}");
            OnGenerationError?.Invoke(e.Message);
        }
        finally
        {
            visionEmbeddings?.Dispose();
            mergedEmbeddings?.Dispose();
            IsGenerating = false;
        }
    }

    private Tensor<float> BuildPromptEmbeddings(string prompt, Texture image, out Tensor<float> visionEmbeddings)
    {
        visionEmbeddings = image != null ? EncodeVision(image) : null;

        if (visionEmbeddings == null)
        {
            Debug.Log("[ModelVLM] TEXT-ONLY MODE (no vision)");
            string fullPrompt = $"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n";
            var tokenIds = _tokenizer.Encode(fullPrompt);
            return EmbedTokens(tokenIds.ToArray());
        }

        Debug.Log($"[ModelVLM] VISION MODE - Vision shape: {visionEmbeddings.shape}");

        string promptPrefix = "<|im_start|>user\n";
        string promptSuffix = $"\n{prompt}<|im_end|>\n<|im_start|>assistant\n";

        var prefixIds = _tokenizer.Encode(promptPrefix);
        var suffixIds = _tokenizer.Encode(promptSuffix);

        using var prefixEmbeddings = EmbedTokens(prefixIds.ToArray());
        using var suffixEmbeddings = EmbedTokens(suffixIds.ToArray());

        Debug.Log($"[ModelVLM] Prefix shape: {prefixEmbeddings.shape}, Suffix shape: {suffixEmbeddings.shape}");

        var result = ConcatenateEmbeddings(prefixEmbeddings, visionEmbeddings, suffixEmbeddings);

        Debug.Log($"[ModelVLM] Merged shape: {result.shape}");

        return result;
    }

    private Tensor<float> EncodeVision(Texture image)
    {
        int targetSize = 256;

        if (image == null)
        {
            Debug.LogWarning("[ModelVLM] No image provided for vision encoding!");
            return null;
        }

        // Convert texture to readable format for Quest 3
        Texture2D readableTexture = null;
        bool needsCleanup = false;

        if (image is RenderTexture)
        {
            RenderTexture rt = image as RenderTexture;
            RenderTexture previous = RenderTexture.active;
            RenderTexture.active = rt;

            readableTexture = new Texture2D(targetSize, targetSize, TextureFormat.RGB24, false);
            readableTexture.ReadPixels(new Rect(0, 0, targetSize, targetSize), 0, 0);
            readableTexture.Apply();

            RenderTexture.active = previous;
            needsCleanup = true;
        }
        else if (image is Texture2D)
        {
            readableTexture = image as Texture2D;
        }
        else
        {
            Debug.LogError($"[ModelVLM] Unsupported texture type: {image.GetType()}");
            return null;
        }

        // Use the correct Sentis API (width, height, channels)
        using var imageTensor = TextureConverter.ToTensor(readableTexture, targetSize, targetSize, 3);

        // Encode vision
        _visionEncoder.SetInput(0, imageTensor);
        _visionEncoder.Schedule();

        Tensor copiedOutput = null;
        _visionEncoder.CopyOutput(0, ref copiedOutput);

        // Cleanup
        if (needsCleanup && readableTexture != null)
        {
            Texture2D.Destroy(readableTexture);
        }

        var result = copiedOutput as Tensor<float>;
        Debug.Log($"[ModelVLM] Vision encoded. Shape: {result.shape}");

        return result;
    }

    private Tensor<float> EmbedTokens(int[] tokenIds)
    {
        using var inputTensor = new Tensor<int>(new TensorShape(1, tokenIds.Length), tokenIds);
        _embedTokens.SetInput(0, inputTensor);
        _embedTokens.Schedule();

        Tensor copiedOutput = null;
        _embedTokens.CopyOutput(0, ref copiedOutput);
        return copiedOutput as Tensor<float>;
    }

    private Tensor<float> ConcatenateEmbeddings(Tensor<float> t1, Tensor<float> t2, Tensor<float> t3)
    {
        int embeddingDimension = t1.shape[2];
        if (t2.shape[2] != embeddingDimension || t3.shape[2] != embeddingDimension)
            throw new InvalidOperationException("Embedding dimensions must match for concatenation.");

        EnsureConcatWorker(embeddingDimension);
        _concatEmbeddings.SetInput(0, t1);
        _concatEmbeddings.SetInput(1, t2);
        _concatEmbeddings.SetInput(2, t3);
        _concatEmbeddings.Schedule();

        Tensor copiedOutput = null;
        _concatEmbeddings.CopyOutput(0, ref copiedOutput);
        return copiedOutput as Tensor<float>;
    }

    private void EnsureConcatWorker(int embeddingDimension)
    {
        if (_concatEmbeddings != null && _concatEmbeddingDimension == embeddingDimension)
            return;

        _concatEmbeddings?.Dispose();
        _concatEmbeddings = null;

        var funcGraph = new FunctionalGraph();
        var inputShape = new DynamicTensorShape(1, -1, embeddingDimension);
        var input1 = funcGraph.AddInput<float>(inputShape);
        var input2 = funcGraph.AddInput<float>(inputShape);
        var input3 = funcGraph.AddInput<float>(inputShape);
        var concatenated = FF.Concat(new[] { input1, input2, input3 }, 1);
        var model = funcGraph.Compile(concatenated);

        _concatEmbeddings = new Worker(model, BACKEND);
        _concatEmbeddingDimension = embeddingDimension;
    }

    private int DecoderPrefill(Tensor<float> embeddings, int sequenceLength, int maxKvSequenceLength)
    {
        _decoder.SetInput("inputs_embeds", embeddings);

        using var positionIds = new Tensor<int>(new TensorShape(1, sequenceLength), BuildRangeArray(sequenceLength, 0));
        _decoder.SetInput("position_ids", positionIds);

        using var attentionMask = new Tensor<int>(new TensorShape(1, sequenceLength), BuildFilledArray(sequenceLength, 1));
        _decoder.SetInput("attention_mask", attentionMask);

        SetEmptyKVCache(maxKvSequenceLength);

        _decoder.Schedule();

        var logits = _decoder.PeekOutput("logits") as Tensor<float>;
        var firstToken = ProcessLogits(logits, sequenceLength - 1);

        UpdateKVCache();
        return firstToken;
    }

    private int DecoderDecode(int tokenId, int position)
    {
        using var embeddings = EmbedTokens(new[] { tokenId });
        _decoder.SetInput("inputs_embeds", embeddings);

        using var positionIds = new Tensor<int>(new TensorShape(1, 1), new[] { position });
        _decoder.SetInput("position_ids", positionIds);

        using var attentionMask = new Tensor<int>(new TensorShape(1, position + 1), BuildFilledArray(position + 1, 1));
        _decoder.SetInput("attention_mask", attentionMask);

        _decoder.Schedule();

        var logits = _decoder.PeekOutput("logits") as Tensor<float>;

        return ProcessLogits(logits, 0);
    }

    private int ProcessLogits(Tensor<float> logits, int index)
    {
        _greedyDecoder.SetInput(0, logits);
        _greedyDecoder.Schedule();

        var argMaxTensor = _greedyDecoder.PeekOutput() as Tensor<int>;
        using var resultTensor = argMaxTensor.ReadbackAndClone();
        return resultTensor[index];
    }

    private void SetEmptyKVCache(int maxKvSequenceLength)
    {
        var shape = new TensorShape(1, NUM_KEY_VALUE_HEADS, 0, HEAD_DIM);
        int maxTensorLength = Math.Max(0, NUM_KEY_VALUE_HEADS * maxKvSequenceLength * HEAD_DIM);

        for (int i = 0; i < MAX_LAYERS; i++)
        {
            _pastKeys[i]?.Dispose();
            _pastValues[i]?.Dispose();

            if (BACKEND == BackendType.GPUCompute && maxTensorLength > 0)
            {
                _pastKeys[i] = new Tensor<float>(shape, new ComputeTensorData(maxTensorLength, clearOnInit: false));
                _pastValues[i] = new Tensor<float>(shape, new ComputeTensorData(maxTensorLength, clearOnInit: false));
            }
            else
            {
                var preallocatedShape = new TensorShape(1, NUM_KEY_VALUE_HEADS, maxKvSequenceLength, HEAD_DIM);
                _pastKeys[i] = new Tensor<float>(preallocatedShape, clearOnInit: false);
                _pastValues[i] = new Tensor<float>(preallocatedShape, clearOnInit: false);
                _pastKeys[i].Reshape(shape);
                _pastValues[i].Reshape(shape);
            }

            _decoder.SetInput($"past_key_values.{i}.key", _pastKeys[i]);
            _decoder.SetInput($"past_key_values.{i}.value", _pastValues[i]);
        }
    }

    private void UpdateKVCache()
    {
        for (int i = 0; i < MAX_LAYERS; i++)
        {
            string keyName = $"present.{i}.key";
            string valueName = $"present.{i}.value";

            Tensor<float> previousKey = _pastKeys[i];
            Tensor<float> previousValue = _pastValues[i];

            Tensor copiedKey = previousKey;
            Tensor copiedValue = previousValue;
            _decoder.CopyOutput(keyName, ref copiedKey);
            _decoder.CopyOutput(valueName, ref copiedValue);

            _pastKeys[i] = copiedKey as Tensor<float>;
            _pastValues[i] = copiedValue as Tensor<float>;

            if (!ReferenceEquals(previousKey, _pastKeys[i]))
                previousKey?.Dispose();
            if (!ReferenceEquals(previousValue, _pastValues[i]))
                previousValue?.Dispose();

            _decoder.SetInput($"past_key_values.{i}.key", _pastKeys[i]);
            _decoder.SetInput($"past_key_values.{i}.value", _pastValues[i]);
        }
    }

    private void ClearKVCache()
    {
        for (int i = 0; i < MAX_LAYERS; i++)
        {
            _pastKeys[i]?.Dispose();
            _pastValues[i]?.Dispose();
            _pastKeys[i] = null;
            _pastValues[i] = null;
        }
    }

    private static int[] BuildRangeArray(int length, int start)
    {
        var values = new int[length];
        for (int i = 0; i < length; i++)
            values[i] = start + i;
        return values;
    }

    private static int[] BuildFilledArray(int length, int value)
    {
        var values = new int[length];
        if (value == 0)
            return values;

        for (int i = 0; i < length; i++)
            values[i] = value;
        return values;
    }

    private void DisposeWorkers()
    {
        _visionEncoder?.Dispose();
        _visionEncoder = null;

        _embedTokens?.Dispose();
        _embedTokens = null;

        _decoder?.Dispose();
        _decoder = null;

        _greedyDecoder?.Dispose();
        _greedyDecoder = null;

        _concatEmbeddings?.Dispose();
        _concatEmbeddings = null;
        _concatEmbeddingDimension = -1;

        ClearKVCache();
    }

    private void OnDestroy()
    {
        DisposeWorkers();
    }
}