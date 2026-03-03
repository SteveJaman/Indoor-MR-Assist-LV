using System;
using System.Runtime.InteropServices;
using UnityEngine;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using System.IO;
using UnityEngine.Networking;

public class ExecuTorchManager : MonoBehaviour
{
    [DllImport("executorch_unity_bridge")]
    private static extern bool LoadModel(string path);

    [DllImport("executorch_unity_bridge")]
    private static extern IntPtr RunInference(IntPtr inputPtr, string prompt);

    [DllImport("executorch_unity_bridge")]
    private static extern bool LoadTokenizer(string path);

    [Header("Demo Visuals")]
    public DiagnosticHUD diagnosticHud;
    public Renderer visionPreviewRenderer;

    [Header("Camera Settings")]
    public int width = 384;
    public int height = 384;

    private WebCamTexture questCamera;
    private NativeArray<byte> inputBuffer;
    private float[] results = new float[1000];
    private Texture2D visionTexture;
    private System.Diagnostics.Stopwatch inferenceTimer = new System.Diagnostics.Stopwatch();
    private bool isNativeLibraryLoaded = false;

    void Start()
    {

#if UNITY_ANDROID && !UNITY_EDITOR
        if (!UnityEngine.Android.Permission.HasUserAuthorizedPermission(UnityEngine.Android.Permission.Camera))
        {
            UnityEngine.Android.Permission.RequestUserPermission(UnityEngine.Android.Permission.Camera);
        }
#endif

        questCamera = new WebCamTexture(width, height);
        questCamera.Play();

        visionTexture = new Texture2D(width, height, TextureFormat.RGBA32, false);
        if (visionPreviewRenderer != null)
            visionPreviewRenderer.material.mainTexture = visionTexture;

        // --- MODEL EXTRACTION LOGIC ---
        string modelName = "fastvlm_vision_vulkan.pte";
        string destinationPath = Path.Combine(Application.persistentDataPath, modelName);

        // --- TOKENIZER EXTRACTION LOGIC ---
        string tokenizerName = "tokenizer.bin";
        string tokenizerDest = Path.Combine(Application.persistentDataPath, tokenizerName);

#if UNITY_ANDROID && !UNITY_EDITOR
        // Extract Model
        if (!File.Exists(destinationPath))
        {
            ExtractFile(modelName, destinationPath);
        } 
        
        // Extract Tokenizer
        if (!File.Exists(tokenizerDest))
        {
            ExtractFile(tokenizerName, tokenizerDest);
        } 
#endif

        try
        {
            if (LoadModel(destinationPath))
            {
                if (LoadTokenizer(tokenizerDest))
                {
                    isNativeLibraryLoaded = true;
                    Debug.Log("ExecuTorch: Model Loaded Successfully.");
                }
                else
                {
                    Debug.LogError("ExecuTorch: Model loaded, but Tokenizer failed!");
                }
            }
        }
        catch (DllNotFoundException e)
        {
            isNativeLibraryLoaded = false;
            Debug.LogError("DLL MISSING: libexecutorch_unity_bridge.so not found! " + e.Message);
        }
    }

    private void ExtractFile(string fileName, string destinationPath)
    {
        Debug.Log($"Extracting {fileName} to {destinationPath}...");
        string sourcePath = Path.Combine(Application.streamingAssetsPath, fileName);

        // For Android, StreamingAssets are inside the compressed APK
        var loadingRequest = UnityWebRequest.Get(sourcePath);
        loadingRequest.SendWebRequest();

        // Block until done (fine for Start() during loading screen)
        while (!loadingRequest.isDone) { }

        if (loadingRequest.result == UnityWebRequest.Result.Success)
        {
            File.WriteAllBytes(destinationPath, loadingRequest.downloadHandler.data);
            Debug.Log($"Successfully extracted {fileName}");
        }
        else
        {
            Debug.LogError($"Failed to extract {fileName}: {loadingRequest.error}");
        }
    }

    void Update()
    {
        if (questCamera.didUpdateThisFrame && questCamera.width > 16)
        {
            if (visionTexture.width != questCamera.width || visionTexture.height != questCamera.height)
            {
                visionTexture.Reinitialize(questCamera.width, questCamera.height);
            }

            Color32[] pixels = questCamera.GetPixels32();

            if (!inputBuffer.IsCreated || inputBuffer.Length != pixels.Length * 4)
            {
                if (inputBuffer.IsCreated) inputBuffer.Dispose();
                inputBuffer = new NativeArray<byte>(pixels.Length * 4, Allocator.Persistent);
            }
            inputBuffer.CopyFrom(MemoryMarshal.Cast<Color32, byte>(pixels).ToArray());

            float elapsedMs = 0;
            string aiDescription = "No Inference"; // Default value

            if (isNativeLibraryLoaded)
            {
                inferenceTimer.Restart();
                unsafe
                {
                    IntPtr ptr = (IntPtr)NativeArrayUnsafeUtility.GetUnsafePtr(inputBuffer);
                    string prompt = "Describe the object in the center of the frame.";

                    // Call the update bridge and convert the pointer to a string immediately
                    IntPtr resultPtr = RunInference(ptr, prompt);
                    aiDescription = Marshal.PtrToStringAnsi(resultPtr);
                }
                inferenceTimer.Stop();
                elapsedMs = (float)inferenceTimer.Elapsed.TotalMilliseconds;
            }

            // Update the HUD and Preview
            if (diagnosticHud != null)
            {
                string status = isNativeLibraryLoaded ? "FastVLM Active" : "DLL MISSING";
                diagnosticHud.UpdateUI(elapsedMs, status, aiDescription);
            }

            visionTexture.SetPixels32(pixels);
            visionTexture.Apply();
        }
    }

    void OnDestroy()
    {
        if (inputBuffer.IsCreated) inputBuffer.Dispose();
    }
}