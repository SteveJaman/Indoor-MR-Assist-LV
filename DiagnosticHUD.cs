using UnityEngine;
using TMPro;

public class DiagnosticHUD : MonoBehaviour
{
    public Transform cameraTransform;
    public TextMeshProUGUI statsText;

    [Header("UI Settings")]
    public float distance = 1.2f; // Distance in front of face
    public Vector3 offset = new Vector3(0.4f, -0.2f, 0); // Bottom-right positioning
    public float followSpeed = 5.0f;

    void Start()
    {
        // 1. Find Camera if slot is empty
        if (cameraTransform == null)
        {
            cameraTransform = Camera.main != null ? Camera.main.transform : FindFirstObjectByType<Camera>()?.transform;
        }

        // 2. Set Canvas to World Space if it isn't already
        Canvas canvas = GetComponent<Canvas>();
        if (canvas != null) canvas.renderMode = RenderMode.WorldSpace;

        // 3. Initial snap so it's not at (0,0,0)
        if (cameraTransform != null)
        {
            SnapToFace();
        }
    }

    void Update()
    {
        if (cameraTransform == null) return;

        // Calculate target position based on camera look direction
        Vector3 targetPos = cameraTransform.position +
                           (cameraTransform.forward * distance) +
                           (cameraTransform.right * offset.x) +
                           (cameraTransform.up * offset.y);

        // Smoothly follow the head movement
        transform.position = Vector3.Lerp(transform.position, targetPos, Time.deltaTime * followSpeed);

        // Always face the user
        transform.LookAt(transform.position + cameraTransform.forward);
    }

    public void SnapToFace()
    {
        if (cameraTransform == null) return;
        transform.position = cameraTransform.position + (cameraTransform.forward * distance);
    }

    public void UpdateUI(float inferenceTime, string topResult, float confidence)
    {
        if (statsText == null) return;

        // Color coding for status
        string statusColor = (topResult.Contains("MISSING") || topResult.Contains("WAITING")) ? "#FF4444" : "#FFFF00";

        statsText.text = $"<color=#00FF00><b>EXECU-TORCH SYSTEM</b></color>\n" +
                         $"<size=80%>----------------------------</size>\n" +
                         $"Status: <color={statusColor}>{topResult}</color>\n" +
                         $"Latency: <color=#00FFFF>{inferenceTime:F1} ms</color>\n" +
                         $"Confidence: {confidence:P1}\n" +
                         $"<size=70%>Target: Quest 3 (Vulkan)</size>";
    }
}