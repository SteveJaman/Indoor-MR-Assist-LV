using UnityEngine;
using TMPro;
public class DiagnosticHUD : MonoBehaviour
{
    public Transform cameraTransform;

    [Header("UI References")]
    // Added these so the script knows what text to update
    public TextMeshProUGUI latencyText;
    public TextMeshProUGUI statusText;
    public TextMeshProUGUI resultText;

    [Header("UI Settings")]
    public float distance = 1.2f;
    public Vector3 offset = new Vector3(0.4f, -0.2f, 0);
    public float followSpeed = 5.0f;

    void Start()
    {
        if (cameraTransform == null)
        {
            cameraTransform = Camera.main != null ? Camera.main.transform : FindFirstObjectByType<Camera>()?.transform;
        }

        Canvas canvas = GetComponent<Canvas>();
        if (canvas != null) canvas.renderMode = RenderMode.WorldSpace;

        if (cameraTransform != null) SnapToFace();
    }

    void Update()
    {
        if (cameraTransform == null) return;

        Vector3 targetPos = cameraTransform.position +
                           (cameraTransform.forward * distance) +
                           (cameraTransform.right * offset.x) +
                           (cameraTransform.up * offset.y);

        transform.position = Vector3.Lerp(transform.position, targetPos, Time.deltaTime * followSpeed);

        // Keeps the HUD facing you
        transform.LookAt(transform.position + cameraTransform.forward);
    }

    public void SnapToFace()
    {
        if (cameraTransform == null) return;
        transform.position = cameraTransform.position + (cameraTransform.forward * distance);
    }
    public void UpdateUI(float latency, string status, string aiResult)
    {
        if (latencyText != null) latencyText.text = $"Latency: {latency:F2}ms";
        if (statusText != null) statusText.text = $"Status: {status}";
        if (resultText != null) resultText.text = $"AI: {aiResult}";
    }
}