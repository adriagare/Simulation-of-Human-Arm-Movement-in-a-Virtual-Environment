using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

/// <summary>
/// Receives marker data over UDP from OptiTrack Python client and visualizes
/// each marker as a small sphere. Spheres are always visible — use them to
/// verify calibration by placing markers on the table edges and checking
/// that they land on the Unity table.
///
/// UDP message format: "frame_id;marker_count;id,x,y,z;id,x,y,z;..."
/// </summary>
public class UDPMarkerReceiver : MonoBehaviour
{
    [Header("UDP Settings")]
    public int listenPort = 5005;

    [Header("Marker Visualization")]
    [Tooltip("Radius of each marker sphere.")]
    public float markerRadius = 0.015f;

    [Header("CSV Logging")]
    [Tooltip("Enable OptiTrack marker CSV logging (starts when headset tracking is active).")]
    public bool enableCSVLog = true;

    [Header("Debug")]
    public bool showDebugLog = false;

    // CSV logging state
    private StreamWriter _csvWriter;
    private string _csvFilePath;
    private bool _csvStarted;

    // Internal state
    private UdpClient _udpClient;
    private Thread _receiveThread;
    private bool _isRunning;

    // Thread-safe data exchange
    private readonly object _lock = new object();
    private string _latestMessage;
    private int _lastFrameId = -1;

    // Latest raw positions — ONLY currently visible markers
    private Dictionary<int, Vector3> _latestRawPositions = new Dictionary<int, Vector3>();

    // Marker sphere GameObjects keyed by marker ID
    private Dictionary<int, GameObject> _markerObjects = new Dictionary<int, GameObject>();

    // Reference to coordinate synchronizer
    private CoordinateSynchronizer _synchronizer;

    private static readonly Color[] MarkerColors = new Color[]
    {
        new Color(1f, 0.3f, 0.3f),   // Red
        new Color(1f, 0.7f, 0.2f),   // Orange
        new Color(0.3f, 1f, 0.3f),   // Green
        new Color(0.3f, 0.3f, 1f),   // Blue
        new Color(1f, 1f, 0.2f),     // Yellow
    };

    // Public status for other scripts to query
    public bool IsReceivingData { get; private set; }
    public int CurrentMarkerCount => _latestRawPositions.Count;

    private float _lastDataTime = 0f;

    void Start()
    {
        _synchronizer = FindFirstObjectByType<CoordinateSynchronizer>();
        StartUDPListener();
    }

    void OnDestroy()
    {
        StopUDPListener();
        ShutdownCSV();
    }

    void OnApplicationQuit()
    {
        ShutdownCSV();
    }

    void Update()
    {
        string message = null;
        lock (_lock)
        {
            message = _latestMessage;
            _latestMessage = null;
        }

        if (message != null)
            ParseMessage(message);

        IsReceivingData = (Time.time - _lastDataTime) < 1.0f;
    }

    private void StartUDPListener()
    {
        _isRunning = true;
        _receiveThread = new Thread(ReceiveLoop) { IsBackground = true };
        _receiveThread.Start();
        Debug.Log($"[UDP] Listening on port {listenPort}");
    }

    private void StopUDPListener()
    {
        _isRunning = false;
        _udpClient?.Close();
        _receiveThread?.Join(500);
    }

    private void ReceiveLoop()
    {
        try
        {
            _udpClient = new UdpClient(listenPort);
            _udpClient.Client.ReceiveTimeout = 500;
            IPEndPoint remoteEP = new IPEndPoint(IPAddress.Any, 0);

            while (_isRunning)
            {
                try
                {
                    byte[] data = _udpClient.Receive(ref remoteEP);
                    string msg = Encoding.UTF8.GetString(data);
                    lock (_lock) { _latestMessage = msg; }
                }
                catch (SocketException) { }
            }
        }
        catch (Exception e)
        {
            if (_isRunning)
                Debug.LogError($"[UDP] Error: {e.Message}");
        }
    }

    private void ParseMessage(string message)
    {
        string[] parts = message.Split(';');
        if (parts.Length < 2) return;

        if (!int.TryParse(parts[0], out int frameId)) return;
        if (!int.TryParse(parts[1], out int markerCount)) return;

        _lastDataTime = Time.time;

        if (showDebugLog && frameId % 100 == 0)
            Debug.Log($"[UDP] Frame {frameId}, {markerCount} markers");

        HashSet<int> activeIds = new HashSet<int>();

        for (int i = 0; i < markerCount && i + 2 < parts.Length; i++)
        {
            string[] mp = parts[i + 2].Split(',');
            if (mp.Length < 4) continue;

            if (!int.TryParse(mp[0], out int markerId)) continue;
            if (!float.TryParse(mp[1], NumberStyles.Float, CultureInfo.InvariantCulture, out float x)) continue;
            if (!float.TryParse(mp[2], NumberStyles.Float, CultureInfo.InvariantCulture, out float y)) continue;
            if (!float.TryParse(mp[3], NumberStyles.Float, CultureInfo.InvariantCulture, out float z)) continue;

            activeIds.Add(markerId);
            Vector3 rawPos = new Vector3(x, y, z);
            _latestRawPositions[markerId] = rawPos;

            // Apply coordinate transform (pre-calibration offset or full transform)
            Vector3 displayPos = _synchronizer != null
                ? _synchronizer.OptiTrackToUnity(rawPos)
                : rawPos;

            // Create or update marker sphere
            if (!_markerObjects.TryGetValue(markerId, out GameObject sphere))
            {
                sphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                sphere.name = $"Marker_{markerId}";
                sphere.transform.localScale = Vector3.one * markerRadius * 2f;
                Destroy(sphere.GetComponent<Collider>());
                int colorIdx = _markerObjects.Count % MarkerColors.Length;
                sphere.GetComponent<Renderer>().material.color = MarkerColors[colorIdx];
                _markerObjects[markerId] = sphere;
            }
            sphere.transform.position = displayPos;
        }

        // Hide spheres for markers no longer visible
        foreach (var kvp in _markerObjects)
            kvp.Value.SetActive(activeIds.Contains(kvp.Key));

        // Log to CSV once headset tracking is active
        if (enableCSVLog && XRPositionLogger.IsTrackingActive)
        {
            if (!_csvStarted) SetupCSV();
            if (_csvWriter != null)
            {
                float t = Time.time;
                for (int i = 0; i < markerCount && i + 2 < parts.Length; i++)
                {
                    string[] mp = parts[i + 2].Split(',');
                    if (mp.Length < 4) continue;
                    _csvWriter.WriteLine($"{t:F4},{frameId},{mp[0]},{mp[1]},{mp[2]},{mp[3]}");
                }
            }
        }

        // Remove stale markers not in current frame
        List<int> staleIds = new List<int>();
        foreach (var kvp in _latestRawPositions)
        {
            if (!activeIds.Contains(kvp.Key))
                staleIds.Add(kvp.Key);
        }
        foreach (int id in staleIds)
            _latestRawPositions.Remove(id);

        _lastFrameId = frameId;
    }

    private void SetupCSV()
    {
        if (!enableCSVLog) return;

        string folder = Path.Combine(Application.dataPath, "..", "Positions", "OptiTrack");
        if (!Directory.Exists(folder))
            Directory.CreateDirectory(folder);

        string ts = DateTime.Now.ToString("yyyy-MM-dd_HH-mm-ss");
        _csvFilePath = Path.Combine(folder, "optitrack_markers_" + ts + ".csv");
        _csvWriter = new StreamWriter(_csvFilePath, false, Encoding.UTF8);
        _csvWriter.AutoFlush = true;
        _csvWriter.WriteLine("time,frame_id,marker_id,pos_x,pos_y,pos_z");
        _csvStarted = true;

        Debug.Log("[UDP] OptiTrack CSV: " + _csvFilePath);
    }

    private void ShutdownCSV()
    {
        if (_csvWriter != null)
        {
            _csvWriter.Flush();
            _csvWriter.Close();
            _csvWriter = null;
        }
    }

    /// <summary>
    /// Returns all currently visible marker IDs and their raw (untransformed) positions.
    /// Only contains markers from the most recent frame.
    /// </summary>
    public Dictionary<int, Vector3> GetAllRawPositions()
    {
        return new Dictionary<int, Vector3>(_latestRawPositions);
    }
}