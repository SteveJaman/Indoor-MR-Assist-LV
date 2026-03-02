using System;
using System.Runtime.InteropServices;
using UnityEngine;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;

public class ExecuTorchManager : MonoBehaviour
{
    // --- NATIVE PLUGIN LINKING ---
    [DllImport("executorch_unity_bridge")]
    private static extern bool LoadModel(string path);

    [DllImport("executorch_unity_bridge")]
    private static extern void RunInference(IntPtr inputPtr, float[] output);

    // --- PROJECT SETTINGS ---
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

    // Safety flag to prevent crashing if the DLL is missing
    private bool isNativeLibraryLoaded = false;

    void Start()
    {
        // 1. Android Permission Request (Required for Quest 3)
#if UNITY_ANDROID && !UNITY_EDITOR
        if (!UnityEngine.Android.Permission.HasUserAuthorizedPermission(UnityEngine.Android.Permission.Camera))
        {
            UnityEngine.Android.Permission.RequestUserPermission(UnityEngine.Android.Permission.Camera);
        }
#endif

        // 2. Setup Camera
        questCamera = new WebCamTexture(width, height);
        questCamera.Play();

        // 3. Setup Texture for the "AI Eyes" Quad
        visionTexture = new Texture2D(width, height, TextureFormat.RGBA32, false);
        if (visionPreviewRenderer != null)
            visionPreviewRenderer.material.mainTexture = visionTexture;

        // 4. Load Model with Safety Catch
        string modelPath = Application.persistentDataPath + "/FastVLM_384.pte";

        try
        {
            if (LoadModel(modelPath))
            {
                isNativeLibraryLoaded = true;
                Debug.Log("ExecuTorch: Vulkan Model Loaded Successfully.");
            }
        }
        catch (DllNotFoundException e)
        {
            isNativeLibraryLoaded = false;
            Debug.LogError("DEMO CRITICAL: libexecutorch_unity_bridge.so not found! HUD will stay in standby. " + e.Message);
        }
    }

    void Update()
    {
        // 1. Only run if the camera actually has a new frame
        if (questCamera.didUpdateThisFrame && questCamera.width > 16)
        {
            // 2. SAFETY CHECK: Resize texture if camera resolution changed
            if (visionTexture.width != questCamera.width || visionTexture.height != questCamera.height)
            {
                Debug.Log($"Resizing Texture to match Camera: {questCamera.width}x{questCamera.height}");
                visionTexture.Reinitialize(questCamera.width, questCamera.height);
            }

            // 3. Get the pixels
            Color32[] pixels = questCamera.GetPixels32();

            // 4. Update the input buffer for the AI
            if (!inputBuffer.IsCreated || inputBuffer.Length != pixels.Length * 4)
            {
                if (inputBuffer.IsCreated) inputBuffer.Dispose();
                inputBuffer = new NativeArray<byte>(pixels.Length * 4, Allocator.Persistent);
            }
            inputBuffer.CopyFrom(MemoryMarshal.Cast<Color32, byte>(pixels).ToArray());

            float elapsedMs = 0;

            // 5. Run AI Inference (Only if DLL is loaded)
            if (isNativeLibraryLoaded)
            {
                inferenceTimer.Restart();
                unsafe
                {
                    IntPtr ptr = (IntPtr)NativeArrayUnsafeUtility.GetUnsafePtr(inputBuffer);
                    RunInference(ptr, results);
                }
                inferenceTimer.Stop();
                elapsedMs = (float)inferenceTimer.Elapsed.TotalMilliseconds;
            }

            // 6. UPDATE HUD
            if (diagnosticHud != null)
            {
                string statusLabel = isNativeLibraryLoaded ? "FastVLM Active" : "DLL MISSING / STANDBY";
                diagnosticHud.UpdateUI(elapsedMs, statusLabel, results[0]);
            }

            // 7. UPDATE VISION QUAD (The part that was crashing)
            visionTexture.SetPixels32(pixels);
            visionTexture.Apply();
        }
    }

    void OnDestroy()
    {
        if (inputBuffer.IsCreated) inputBuffer.Dispose();
    }
}