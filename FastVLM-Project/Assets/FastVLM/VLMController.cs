using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Video;
using System.Collections;

public class VLMController : MonoBehaviour
{
    [Header("Model")]
    [SerializeField] private ModelVLM vlmModel;

    [Header("UI Components")]
    [SerializeField] private RawImage imageDisplay;
    [SerializeField] private Text outputText;
    [SerializeField] private InputField promptInput;

    [Header("Video & Audio Settings")]
    [SerializeField] private VideoPlayer videoPlayer;
    [SerializeField] private VideoClip videoClipAsset;
    [SerializeField] private bool muteAudio = true;

    [Header("Research Settings")]
    [SerializeField][TextArea(3, 10)] private string researchPrompt = "Describe what you see in one sentence.";
    [SerializeField] private float analysisCooldown = 3.0f;

    [Header("Quest 3 Debug")]
    [SerializeField] private bool enableDebugLogs = true;

    private RenderTexture videoRenderTexture;
    private Texture2D readbackTexture;
    private bool isDestroying;
    private bool isReadingTexture;

    private void Start()
    {
        if (promptInput != null) promptInput.text = researchPrompt;

        Canvas canvas = GetComponentInParent<Canvas>();
        if (canvas != null)
        {
            if (canvas.renderMode != RenderMode.WorldSpace)
            {
                canvas.renderMode = RenderMode.WorldSpace;
                if (enableDebugLogs) Debug.Log("[VLM] Canvas set to WorldSpace mode");
            }
            canvas.gameObject.SetActive(true);
        }

        if (imageDisplay == null) Debug.LogWarning("[VLM] imageDisplay RawImage is not assigned!");
        if (outputText == null) Debug.LogWarning("[VLM] outputText Text is not assigned!");
        if (videoPlayer == null) Debug.LogWarning("[VLM] videoPlayer is not assigned!");

        vlmModel.OnTokenGenerated += (t) => SetOutputText(t);
        vlmModel.OnGenerationComplete += (t, c, ms) => SetOutputText(t);
        vlmModel.OnGenerationError += (e) => SetOutputText("AI Error: " + e);

        StartCoroutine(InitializeSequence());
    }

    private IEnumerator InitializeSequence()
    {
        SetOutputText("Step 1: Video & Audio Init...");
        SetupVideoSource();

        float timeout = 5f;
        float elapsed = 0f;
        while (!videoPlayer.isPrepared && elapsed < timeout)
        {
            elapsed += Time.deltaTime;
            yield return null;
        }

        if (videoPlayer.isPrepared)
        {
            videoPlayer.Play();
            while (videoPlayer.frame <= 0) yield return null;

            if (imageDisplay != null)
            {
                imageDisplay.texture = videoRenderTexture;
                imageDisplay.SetAllDirty();
            }
        }

        SetOutputText("Step 2: AI Memory Allocation...");
        var initTask = vlmModel.Initialize();
        while (!initTask.IsCompleted) yield return null;

        if (!vlmModel.IsInitialized)
        {
            SetOutputText("ERROR: AI failed, but video is live!");
            yield break;
        }

        SetOutputText("System Online. Starting Analysis...");
        StartCoroutine(AutoInferenceLoop());
    }

    private void SetupVideoSource()
    {
        if (videoPlayer == null) videoPlayer = GetComponent<VideoPlayer>();

        videoPlayer.source = VideoSource.VideoClip;
        videoPlayer.clip = videoClipAsset;
        videoPlayer.renderMode = VideoRenderMode.RenderTexture;

        if (videoRenderTexture == null)
        {
            videoRenderTexture = new RenderTexture(256, 256, 0, RenderTextureFormat.ARGB32);
            videoRenderTexture.filterMode = FilterMode.Bilinear;
            videoRenderTexture.Create();
        }

        videoPlayer.targetTexture = videoRenderTexture;

        if (readbackTexture == null)
        {
            readbackTexture = new Texture2D(256, 256, TextureFormat.RGB24, false);
            readbackTexture.filterMode = FilterMode.Bilinear;
        }

        videoPlayer.audioOutputMode = VideoAudioOutputMode.Direct;
        videoPlayer.controlledAudioTrackCount = 1;
        videoPlayer.Prepare();
    }

    private IEnumerator AutoInferenceLoop()
    {
        while (!isDestroying)
        {
            if (vlmModel.IsInitialized && !vlmModel.IsGenerating && videoPlayer.isPlaying && !isReadingTexture)
            {
                string currentPrompt = promptInput != null ? promptInput.text : researchPrompt;
                SetOutputText("FastVLM Thinking...");

                yield return StartCoroutine(CaptureAndAnalyze(currentPrompt));

                yield return new WaitForSeconds(analysisCooldown);
            }
            yield return null;
        }
    }

    private IEnumerator CaptureAndAnalyze(string prompt)
    {
        isReadingTexture = true;
        yield return new WaitForEndOfFrame();

        if (!videoPlayer.isPlaying)
        {
            isReadingTexture = false;
            yield break;
        }

        System.Threading.Tasks.Task genTask = null;

        try
        {
            RenderTexture previousActive = RenderTexture.active;
            RenderTexture.active = videoRenderTexture;
            readbackTexture.ReadPixels(new Rect(0, 0, 256, 256), 0, 0, false);
            readbackTexture.Apply(false);
            RenderTexture.active = previousActive;

            if (enableDebugLogs) Debug.Log($"[VLM] Texture captured. Starting inference...");

            // Start the task inside the try block
            genTask = vlmModel.GenerateFromPrompt(prompt, readbackTexture);
        }
        catch (System.Exception e)
        {
            Debug.LogError($"[VLM] Capture Error: {e.Message}");
            SetOutputText($"Capture Error: {e.Message}");
            isReadingTexture = false;
            yield break; // Exit early if the capture failed
        }

        // WAIT for the task OUTSIDE the try/catch block (Fixes CS1626)
        if (genTask != null)
        {
            while (!genTask.IsCompleted)
            {
                yield return null;
            }
        }

        if (enableDebugLogs) Debug.Log("[VLM] Inference complete");
        isReadingTexture = false;
    }

    private void SetOutputText(string text)
    {
        if (outputText != null) outputText.text = text;
    }

    private void OnDestroy()
    {
        isDestroying = true;
        if (videoRenderTexture != null) { videoRenderTexture.Release(); Destroy(videoRenderTexture); }
        if (readbackTexture != null) { Destroy(readbackTexture); }
    }
}