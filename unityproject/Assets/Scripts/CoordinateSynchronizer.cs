using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;
using UnityEngine.InputSystem;

/// <summary>
/// Press C with 4 markers on the table corners to:
///   1. Save the raw OptiTrack coordinates to a CSV (table verification)
///   2. Compute the OptiTrack → Unity coordinate transform (needed for arm)
///
/// Before calibration, a rough offset is applied so markers appear near the
/// table instead of at the floor. Adjust preCalibrationOffset if needed.
/// </summary>
public class CoordinateSynchronizer : MonoBehaviour
{
    [Header("Table Corners (Unity World Space)")]
    [Tooltip("Default values assume table surface Y=1.05. Override per scene from the Inspector if the table position changes; keep all four corners coplanar.")]
    public Vector3 unityPoint1 = new Vector3( 0.7f, 1.10f, 0.365f);    // bottom-right (closest to user)
    public Vector3 unityPoint2 = new Vector3(-0.7f, 1.10f, 0.365f);    // bottom-left  (closest to user)
    public Vector3 unityPoint3 = new Vector3( 0.7f, 1.10f, 1.035f);    // top-right    (far from user)
    public Vector3 unityPoint4 = new Vector3(-0.7f, 1.10f, 1.035f);    // top-left     (far from user)

    [Header("Pre-Calibration Settings")]
    [Tooltip("OptiTrack origin (0,0,0) sits at the table's bottom-right corner. " +
             "OptiTrack +X points LEFT (Unity -X), so X must be negated.")]
    public bool negateX = true;

    [Header("Calibration Key")]
    public Key captureKey = Key.C;

    [Header("Status")]
    [SerializeField] private bool isCalibrated = false;

    public bool IsCalibrated => isCalibrated;

    private Matrix4x4 _transformMatrix = Matrix4x4.identity;
    private UDPMarkerReceiver _receiver;

    void Start()
    {
        _receiver = FindFirstObjectByType<UDPMarkerReceiver>();
        if (_receiver == null)
            Debug.LogError("[Calibration] UDPMarkerReceiver not found!");

        Debug.Log("[Calibration] Place 4 markers on table corners and press C.");
    }

    void Update()
    {
        if (isCalibrated) return;
        if (Keyboard.current != null && Keyboard.current[captureKey].wasPressedThisFrame)
            CaptureCalibrationPoints();
    }

    public Vector3 OptiTrackToUnity(Vector3 optiTrackPos)
    {
        if (!isCalibrated)
        {
            // OptiTrack origin (0,0,0) = bottom-right corner = unityPoint1
            // OptiTrack +X goes LEFT, Unity +X goes RIGHT → negate X
            float x = negateX ? -optiTrackPos.x : optiTrackPos.x;
            return new Vector3(
                x + unityPoint1.x,
                optiTrackPos.y + unityPoint1.y,
                optiTrackPos.z + unityPoint1.z
            );
        }

        // Negate X to match handedness (same flip applied during calibration)
        Vector3 flipped = new Vector3(-optiTrackPos.x, optiTrackPos.y, optiTrackPos.z);
        return _transformMatrix.MultiplyPoint3x4(flipped);
    }

    private void CaptureCalibrationPoints()
    {
        if (_receiver == null) return;

        Dictionary<int, Vector3> allMarkers = _receiver.GetAllRawPositions();

        if (allMarkers.Count < 4)
        {
            Debug.LogWarning($"[Calibration] Need 4 markers, only {allMarkers.Count} visible.");
            return;
        }
        if (allMarkers.Count > 4)
        {
            Debug.LogWarning($"[Calibration] {allMarkers.Count} markers visible, need exactly 4.");
            return;
        }

        // Sort markers geometrically to assign to table corners.
        // OptiTrack layout: origin at bottom-right, +X goes LEFT, +Z goes away.
        // Split into right-side (low X) and left-side (high X), then by Z.
        var markers = new List<(int id, Vector3 pos)>();
        foreach (var kvp in allMarkers)
            markers.Add((kvp.Key, kvp.Value));

        markers.Sort((a, b) => a.pos.x.CompareTo(b.pos.x));

        var rightPair = new List<(int id, Vector3 pos)> { markers[0], markers[1] };
        var leftPair  = new List<(int id, Vector3 pos)> { markers[2], markers[3] };
        rightPair.Sort((a, b) => a.pos.z.CompareTo(b.pos.z));
        leftPair.Sort((a, b) => a.pos.z.CompareTo(b.pos.z));

        int[] markerIds = new int[4];
        Vector3[] optiPoints = new Vector3[4];

        markerIds[0] = rightPair[0].id; optiPoints[0] = rightPair[0].pos; // bottom-right → unityPoint1
        markerIds[1] = leftPair[0].id;  optiPoints[1] = leftPair[0].pos;  // bottom-left  → unityPoint2
        markerIds[2] = rightPair[1].id; optiPoints[2] = rightPair[1].pos; // top-right    → unityPoint3
        markerIds[3] = leftPair[1].id;  optiPoints[3] = leftPair[1].pos;  // top-left     → unityPoint4

        Vector3[] unityPoints = { unityPoint1, unityPoint2, unityPoint3, unityPoint4 };

        SaveCalibrationCSV(markerIds, optiPoints, unityPoints);

        // Negate X before computing the rigid transform.
        // OptiTrack is right-handed, Unity is left-handed. A rigid rotation
        // cannot change handedness, so negating X here makes both coordinate
        // systems same-handed. The transform then only needs rotation + translation.
        Vector3[] optiFlipped = new Vector3[4];
        for (int i = 0; i < 4; i++)
            optiFlipped[i] = new Vector3(-optiPoints[i].x, optiPoints[i].y, optiPoints[i].z);

        _transformMatrix = ComputeRigidTransform(optiFlipped, unityPoints);
        isCalibrated = true;

        // Log results
        string[] names = { "bottom-right", "bottom-left", "top-right", "top-left" };
        float totalError = 0f;
        for (int i = 0; i < 4; i++)
        {
            Vector3 t = _transformMatrix.MultiplyPoint3x4(optiFlipped[i]);
            float err = Vector3.Distance(t, unityPoints[i]);
            totalError += err;
            Debug.Log($"  Marker {markerIds[i]} → {names[i]}: error {err * 1000f:F2} mm");
        }

        float meanError = totalError / 4f;
        Debug.Log($"[Calibration] Done! Mean error: {meanError * 1000f:F2} mm");

        if (meanError > 0.05f)
            Debug.LogWarning("[Calibration] Mean error > 50mm — check marker placement.");
    }

    private void SaveCalibrationCSV(int[] ids, Vector3[] optiPositions, Vector3[] unityPositions)
    {
        string folder = Path.Combine(Application.dataPath, "..", "Positions", "Calibration");
        if (!Directory.Exists(folder))
            Directory.CreateDirectory(folder);

        string ts = DateTime.Now.ToString("yyyy-MM-dd_HH-mm-ss");
        string path = Path.Combine(folder, $"calibration_{ts}.csv");

        using (StreamWriter w = new StreamWriter(path, false, Encoding.UTF8))
        {
            w.WriteLine("marker_id,opti_x,opti_y,opti_z,unity_x,unity_y,unity_z,corner_name");
            string[] names = { "bottom-right", "bottom-left", "top-right", "top-left" };

            for (int i = 0; i < 4; i++)
            {
                Vector3 o = optiPositions[i];
                w.WriteLine($"{ids[i]},{o.x:F6},{o.y:F6},{o.z:F6},,,,raw_capture_{i}");
            }

            for (int i = 0; i < 4; i++)
            {
                Vector3 u = unityPositions[i];
                w.WriteLine($",,,,{u.x:F6},{u.y:F6},{u.z:F6},{names[i]}_expected");
            }
        }

        Debug.Log($"[Calibration] Corner coordinates saved to: {path}");
    }

    // ─── Transform computation ──────────────────────────────────────

    private Matrix4x4 ComputeRigidTransform(Vector3[] src, Vector3[] dst)
    {
        int n = src.Length;

        Vector3 srcC = Vector3.zero, dstC = Vector3.zero;
        for (int i = 0; i < n; i++) { srcC += src[i]; dstC += dst[i]; }
        srcC /= n; dstC /= n;

        Vector3[] sc = new Vector3[n], dc = new Vector3[n];
        for (int i = 0; i < n; i++) { sc[i] = src[i] - srcC; dc[i] = dst[i] - dstC; }

        Vector3 se1 = (sc[1] - sc[0]).normalized;
        Vector3 se3 = Vector3.Cross(se1, (sc[2] - sc[0]).normalized).normalized;
        if (se3.y < 0) se3 = -se3; // vertical axis must point UP for coplanar table calibration
        Vector3 se2 = Vector3.Cross(se3, se1).normalized;

        Vector3 de1 = (dc[1] - dc[0]).normalized;
        Vector3 de3 = Vector3.Cross(de1, (dc[2] - dc[0]).normalized).normalized;
        if (de3.y < 0) de3 = -de3; // vertical axis must point UP for coplanar table calibration
        Vector3 de2 = Vector3.Cross(de3, de1).normalized;

        if (se3.sqrMagnitude < 0.001f || de3.sqrMagnitude < 0.001f)
            return Matrix4x4.identity;

        Matrix4x4 sB = Matrix4x4.identity, dB = Matrix4x4.identity;
        sB.SetColumn(0, se1); sB.SetColumn(1, se2); sB.SetColumn(2, se3);
        dB.SetColumn(0, de1); dB.SetColumn(1, de2); dB.SetColumn(2, de3);

        Matrix4x4 rot = dB * sB.transpose;
        Vector3 trans = dstC - rot.MultiplyPoint3x4(srcC);

        Matrix4x4 r = Matrix4x4.identity;
        r.SetColumn(0, rot.GetColumn(0));
        r.SetColumn(1, rot.GetColumn(1));
        r.SetColumn(2, rot.GetColumn(2));
        r.SetColumn(3, new Vector4(trans.x, trans.y, trans.z, 1f));
        return r;
    }

}
