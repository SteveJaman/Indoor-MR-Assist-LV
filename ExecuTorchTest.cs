using System.Runtime.InteropServices;
using UnityEngine;

public class ExecuTorchTest : MonoBehaviour
{
    // Point to the name of your library (without 'lib' or '.so')
    private const string PluginName = "executorch";

    // We'll call a simple internal logging function to see if it's alive
    [DllImport(PluginName)]
    public static extern void et_log_set_level(int level);

    void Start()
    {
        Debug.Log("Attempting to wake up ExecuTorch...");
        try 
        {
            // Set log level to 1 (Info)
            et_log_set_level(1);
            Debug.Log("<color=green>SUCCESS:</color> ExecuTorch library linked and responding!");
        } 
        catch (System.DllNotFoundException e) 
        {
            Debug.LogError("FAILED: Library not found. Check the folder path and Inspector settings! " + e.Message);
        }
    }
}