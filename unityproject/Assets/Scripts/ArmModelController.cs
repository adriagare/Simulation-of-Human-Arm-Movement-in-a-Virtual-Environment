using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;

/// <summary>
/// First-person virtual arm driven by 3 OptiTrack markers (shoulder, elbow, hand).
///
/// Press O to activate after calibration.
///
/// If armModelPrefab is assigned (drag Y Bot FBX), a realistic skinned arm is shown.
/// Otherwise falls back to capsule/sphere primitives.
///
/// Marker identification: shoulder = closest to headset, hand = farthest from shoulder.
/// Marker loss: joints hold last position; arm hides after markerLossTimeout with 0 markers.
/// </summary>
public class ArmModelController : MonoBehaviour
{
    [Header("Activation")]
    public Key activateKey = Key.O;

    [Header("Arm Model")]
    [Tooltip("Drag the Y Bot FBX (or any Mixamo humanoid) here. Leave empty for primitive fallback.")]
    public GameObject armModelPrefab;
    [Tooltip("Scale of the arm model (reduce if arm appears too large)")]
    public float modelScale = 0.85f;

    [Header("Fallback Primitives")]
    public float armRadius = 0.04f;
    public float handRadius = 0.05f;
    public Color upperArmColor = new Color(0.85f, 0.72f, 0.6f);
    public Color forearmColor  = new Color(0.82f, 0.69f, 0.57f);
    public Color handColor     = new Color(0.88f, 0.75f, 0.63f);

    [Header("Smoothing")]
    [Tooltip("Position smoothing for joint targets. Lower = smoother but laggier.")]
    public float smoothSpeed = 12f;
    [Tooltip("Rotation smoothing for the arm bones. Lower values kill residual twist from marker jitter.")]
    public float rotationSmoothSpeed = 10f;

    [Header("Marker Loss")]
    public float markerLossTimeout = 2.0f;

    [Header("Debug")]
    [Tooltip("Disabled by code regardless of this value — the spheres pollute the first-person view during the experiment.")]
    public bool showJointSpheres = false;

    [Header("Status (read-only)")]
    [SerializeField] private bool armModeActive = false;
    [SerializeField] private int visibleMarkers = 0;

    public bool IsArmModeActive => armModeActive;
    public bool IsTracking => armModeActive && _initialized;
    public Vector3 HandPositionUnity     => _jointSmoothed[HAND];
    public Vector3 ShoulderPositionUnity => _jointSmoothed[SHOULDER];
    public Vector3 ElbowPositionUnity    => _jointSmoothed[ELBOW];
    public bool IsPlaybackActive => _playbackActive;

    // Marker-derived joints kept fresh even when playback overrides the render.
    // The displayed arm follows _jointSmoothed; these buffers track what the
    // user is physically doing, so callers can read the real hand for hit
    // detection or analytical perturbation pivots while the render is hijacked.
    private Vector3[] _realJoints       = new Vector3[3];
    private bool      _realInitialized;
    public Vector3 RealHandPositionUnity     => _realInitialized ? _realJoints[HAND]     : _jointSmoothed[HAND];
    public Vector3 RealShoulderPositionUnity => _realInitialized ? _realJoints[SHOULDER] : _jointSmoothed[SHOULDER];
    public Vector3 RealElbowPositionUnity    => _realInitialized ? _realJoints[ELBOW]    : _jointSmoothed[ELBOW];
    public bool HasRealHand => _realInitialized;

    // Playback override (driven by NN) — when active, marker input is ignored
    // and the hand is moved to the provided position each frame. Elbow is
    // resolved via simple 2-link IK from the frozen shoulder.
    private bool    _playbackActive;
    private Vector3 _playbackShoulder;
    private float   _playbackUpperArmLen;
    private float   _playbackForearmLen;

    public void BeginPlayback()
    {
        if (!_initialized) return;
        _playbackShoulder   = _jointSmoothed[SHOULDER];
        _playbackUpperArmLen = Mathf.Max(0.05f, Vector3.Distance(_jointSmoothed[SHOULDER], _jointSmoothed[ELBOW]));
        _playbackForearmLen  = Mathf.Max(0.05f, Vector3.Distance(_jointSmoothed[ELBOW],    _jointSmoothed[HAND]));
        _playbackActive = true;
    }

    public void EndPlayback()
    {
        _playbackActive = false;
        _markersLost = false;
        _boneRotInitialized = false;
        // Hand the render buffers back to the marker-derived positions so the
        // displayed arm doesn't jump from the perturbed position to the real
        // one when playback releases. The real-joint buffer has been kept
        // fresh throughout the playback window.
        if (_realInitialized)
        {
            for (int j = 0; j < 3; j++)
            {
                _jointTargets[j]  = _realJoints[j];
                _jointSmoothed[j] = _realJoints[j];
            }
            _initialized = true;
        }
        else
        {
            _initialized = false;
        }
    }

    public void SetPlaybackHand(Vector3 handPos)
    {
        if (!_playbackActive) return;
        // Shoulder and elbow follow the live markers so the visible arm
        // translates with the participant; only the wrist (hand) is
        // displaced by the perturbation. This avoids the rendered arm
        // drifting away from the real body when the participant shifts.
        _jointTargets[SHOULDER] = _realInitialized ? _realJoints[SHOULDER] : _playbackShoulder;
        _jointTargets[ELBOW]    = _realInitialized ? _realJoints[ELBOW]
                                                   : ResolveElbowIK(_playbackShoulder, handPos,
                                                                    _playbackUpperArmLen, _playbackForearmLen);
        _jointTargets[HAND]     = handPos;
    }

    private Vector3 ResolveElbowIK(Vector3 shoulder, Vector3 hand, float l1, float l2)
    {
        Vector3 diff = hand - shoulder;
        float d = diff.magnitude;
        if (d < 1e-4f) return shoulder + Vector3.down * l1;

        // Clamp so triangle is feasible
        d = Mathf.Clamp(d, Mathf.Abs(l1 - l2) + 1e-3f, l1 + l2 - 1e-3f);
        Vector3 axis = diff / d;

        // Cosine rule: distance along axis from shoulder to elbow-projection
        float a = (l1 * l1 - l2 * l2 + d * d) / (2f * d);
        float h = Mathf.Sqrt(Mathf.Max(0f, l1 * l1 - a * a));

        // Bend axis: prefer downward deflection (natural elbow)
        Vector3 bend = Vector3.ProjectOnPlane(Vector3.down, axis).normalized;
        if (bend.sqrMagnitude < 1e-4f) bend = Vector3.ProjectOnPlane(Vector3.forward, axis).normalized;

        return shoulder + axis * a + bend * h;
    }

    private const int SHOULDER = 0, ELBOW = 1, HAND = 2;

    // Anatomical offset applied at the render stage to translate the
    // shoulder marker into the underlying joint centre. Currently disabled
    // (zero) because combined with the forced hand/elbow bone snap below
    // it stretched the upper-arm mesh past what the skinning could absorb,
    // producing a visible gap between the upper arm and the forearm. If a
    // visual offset is re-enabled, the bone scaling logic also needs to
    // adapt the upper-arm length to keep the elbow in the right place.
    private static readonly Vector3 SHOULDER_OFFSET_LOCAL = Vector3.zero;

    private UDPMarkerReceiver _receiver;
    private CoordinateSynchronizer _synchronizer;
    private Transform _headset;

    private Vector3[] _jointTargets  = new Vector3[3];
    private Vector3[] _jointSmoothed = new Vector3[3];
    private float[]   _jointLastSeen = new float[3];
    private bool  _initialized;
    private bool  _markersLost;
    private float _lostSince;
    private float _allLostTimer;
    private List<Vector3> _markerPositions = new List<Vector3>();

    // ─── Rigged model state ─────────────────────────────────────────
    private bool _useModel;
    private GameObject _modelInstance;
    private Transform _upperArmBone;   // mixamorig:RightArm
    private Transform _forearmBone;    // mixamorig:RightForeArm
    private Transform _handBone;       // mixamorig:RightHand
    private Vector3    _shoulderBindOffset;
    private Quaternion _upperArmBindLocalRot;
    private Quaternion _forearmBindLocalRot;
    private Vector3    _handBindLocalPos;
    // Local axis of each bone that was closest to world-up at bind. Used to
    // resolve the twist (roll) ambiguity of Quaternion.FromToRotation, so
    // the forearm/hand keep a "palm-down" orientation as the arm moves laterally.
    private Vector3    _upperArmRollAxisLocal = Vector3.up;
    private Vector3    _forearmRollAxisLocal  = Vector3.up;
    private Quaternion _upperArmSmoothedRot;
    private Quaternion _forearmSmoothedRot;
    private bool       _boneRotInitialized;

    // ─── Fallback primitives ────────────────────────────────────────
    private GameObject _primUpperArm, _primForearm, _primHand;

    // ─── Joint debug spheres ────────────────────────────────────────
    private GameObject _shoulderSphere, _elbowSphere, _handSphere;

    // ================================================================
    //  Lifecycle
    // ================================================================

    void Start()
    {
        // Force-disable the debug spheres regardless of any serialized
        // Inspector value. They are visually noisy in first-person VR and
        // we never want them on during an experiment session.
        showJointSpheres = false;

        // Defensive cleanup: destroy any leftover joint sphere GameObjects
        // that may have been saved into the scene during prior sessions.
        for (int i = transform.childCount - 1; i >= 0; i--)
        {
            Transform c = transform.GetChild(i);
            if (c.name.StartsWith("Joint_")) Destroy(c.gameObject);
        }

        _receiver     = FindFirstObjectByType<UDPMarkerReceiver>();
        _synchronizer = FindFirstObjectByType<CoordinateSynchronizer>();

        Camera cam = Camera.main;
        if (cam != null) _headset = cam.transform;

        if (_receiver == null) Debug.LogError("[Arm] UDPMarkerReceiver not found!");
        if (_headset  == null) Debug.LogError("[Arm] Main Camera not found!");

        // Try rigged model first, fall back to primitives
        if (armModelPrefab != null)
        {
            _useModel = SetupRiggedModel();
            if (!_useModel)
                Debug.LogWarning("[Arm] Bone search failed — using primitive shapes.");
        }
        if (!_useModel)
            CreatePrimitives();

        CreateJointSpheres();
        SetVisualsActive(false);
    }

    void Update()
    {
        if (Keyboard.current != null && Keyboard.current[activateKey].wasPressedThisFrame)
            ToggleArmMode();

        if (!armModeActive || _receiver == null || _headset == null) return;

        // ── Playback override: marker logic still runs in parallel so
        //    RealHandPositionUnity stays fresh while the render is hijacked. ──
        if (_playbackActive)
        {
            UpdateRealJointsFromMarkers();
            float tp = smoothSpeed > 0 ? Time.deltaTime * smoothSpeed : 1f;
            for (int j = 0; j < 3; j++)
                _jointSmoothed[j] = Vector3.Lerp(_jointSmoothed[j], _jointTargets[j], tp);
            SetVisualsActive(true);
            if (_useModel) UpdateRiggedModel(); else UpdatePrimitives();
            UpdateJointSpheres();
            return;
        }

        // ── Gather markers, transform to Unity space ────────────────
        Dictionary<int, Vector3> allRaw = _receiver.GetAllRawPositions();
        _markerPositions.Clear();
        foreach (var kvp in allRaw)
            _markerPositions.Add(TransformPoint(kvp.Value));

        // Keep only 3 closest to headset
        if (_markerPositions.Count > 3)
        {
            _markerPositions.Sort((a, b) =>
                Vector3.SqrMagnitude(a - _headset.position)
                    .CompareTo(Vector3.SqrMagnitude(b - _headset.position)));
            _markerPositions.RemoveRange(3, _markerPositions.Count - 3);
        }

        visibleMarkers = _markerPositions.Count;
        float now = Time.time;

        // ── Marker identification / tracking ────────────────────────
        if (_markerPositions.Count >= 3)
        {
            _allLostTimer = 0f;

            if (!_initialized)
            {
                // First detection — identify joints from scratch
                IdentifyByHeadset(_markerPositions);
                for (int j = 0; j < 3; j++)
                {
                    _jointSmoothed[j] = _jointTargets[j];
                    _jointLastSeen[j] = now;
                }
                _initialized = true;
                _markersLost = false;
            }
            else if (_markersLost)
            {
                // Recovering from loss
                float lostDuration = now - _lostSince;
                if (lostDuration > 0.3f)
                {
                    // Long loss (>300ms): markers may have moved significantly,
                    // re-identify from scratch and snap to new positions
                    IdentifyByHeadset(_markerPositions);
                    for (int j = 0; j < 3; j++)
                    {
                        _jointSmoothed[j] = _jointTargets[j];
                        _jointLastSeen[j] = now;
                    }
                }
                else
                {
                    // Brief flicker: continue tracking normally, let
                    // smoothing absorb any small jump
                    MatchMarkersToJoints(_markerPositions, now);
                    for (int j = 0; j < 3; j++)
                        _jointLastSeen[j] = now;
                }
                _markersLost = false;
            }
            else
            {
                MatchMarkersToJoints(_markerPositions, now);
                for (int j = 0; j < 3; j++)
                    _jointLastSeen[j] = now;
            }
        }
        else if (_initialized)
        {
            // Fewer than 3 markers — freeze arm at last good position
            if (!_markersLost)
            {
                _markersLost = true;
                _lostSince = now;
            }

            if (_markerPositions.Count == 0)
                _allLostTimer += Time.deltaTime;
            else
                _allLostTimer = 0f;

            if (_allLostTimer > markerLossTimeout)
            {
                SetVisualsActive(false);
                return;
            }
        }
        else
        {
            SetVisualsActive(false);
            return;
        }

        // ── Smooth ──────────────────────────────────────────────────
        float t = smoothSpeed > 0 ? Time.deltaTime * smoothSpeed : 1f;
        for (int j = 0; j < 3; j++)
            _jointSmoothed[j] = Vector3.Lerp(_jointSmoothed[j], _jointTargets[j], t);

        // Mirror to the real-joint buffer so the API stays consistent
        // whether or not playback is currently overriding the render.
        for (int j = 0; j < 3; j++) _realJoints[j] = _jointSmoothed[j];
        _realInitialized = true;

        // ── Render ──────────────────────────────────────────────────
        SetVisualsActive(true);

        if (_useModel)
            UpdateRiggedModel();
        else
            UpdatePrimitives();

        UpdateJointSpheres();
    }

    // ================================================================
    //  Real-marker tracking during playback
    // ================================================================

    // Lightweight parallel of the main marker pipeline: updates _realJoints
    // from the raw OptiTrack feed while the visible arm is driven by the
    // playback override. Identification uses headset proximity on first
    // contact, then nearest-joint matching on subsequent frames.
    private void UpdateRealJointsFromMarkers()
    {
        if (_receiver == null || _headset == null) return;

        Dictionary<int, Vector3> allRaw = _receiver.GetAllRawPositions();
        var pts = new List<Vector3>(allRaw.Count);
        foreach (var kvp in allRaw) pts.Add(TransformPoint(kvp.Value));

        if (pts.Count > 3)
        {
            pts.Sort((a, b) =>
                Vector3.SqrMagnitude(a - _headset.position)
                    .CompareTo(Vector3.SqrMagnitude(b - _headset.position)));
            pts.RemoveRange(3, pts.Count - 3);
        }

        if (pts.Count < 3) return;  // hold last good values

        if (!_realInitialized)
        {
            // First contact during playback: identify by headset proximity.
            Vector3 head = _headset.position;
            int sIdx = 0; float minD = float.MaxValue;
            for (int i = 0; i < 3; i++)
            {
                float d = Vector3.SqrMagnitude(pts[i] - head);
                if (d < minD) { minD = d; sIdx = i; }
            }
            int hIdx = -1; float maxD = -1f;
            for (int i = 0; i < 3; i++)
            {
                if (i == sIdx) continue;
                float d = Vector3.SqrMagnitude(pts[i] - pts[sIdx]);
                if (d > maxD) { maxD = d; hIdx = i; }
            }
            int eIdx = 3 - sIdx - hIdx;
            _realJoints[SHOULDER] = pts[sIdx];
            _realJoints[ELBOW]    = pts[eIdx];
            _realJoints[HAND]     = pts[hIdx];
            _realInitialized = true;
            return;
        }

        // Continuous frames: assign each marker to its nearest joint slot.
        bool[] jointTaken = new bool[3];
        bool[] markerUsed = new bool[3];
        for (int pass = 0; pass < 3; pass++)
        {
            float bestDist = float.MaxValue;
            int bestM = -1, bestJ = -1;
            for (int m = 0; m < 3; m++)
            {
                if (markerUsed[m]) continue;
                for (int j = 0; j < 3; j++)
                {
                    if (jointTaken[j]) continue;
                    float d = Vector3.SqrMagnitude(pts[m] - _realJoints[j]);
                    if (d < bestDist) { bestDist = d; bestM = m; bestJ = j; }
                }
            }
            if (bestM >= 0)
            {
                _realJoints[bestJ] = pts[bestM];
                jointTaken[bestJ] = true;
                markerUsed[bestM] = true;
            }
        }
    }

    // ================================================================
    //  Marker identification (unchanged)
    // ================================================================

    private void IdentifyByHeadset(List<Vector3> markers)
    {
        Vector3 headPos = _headset.position;

        int sIdx = 0;
        float minD = float.MaxValue;
        for (int i = 0; i < 3; i++)
        {
            float d = Vector3.SqrMagnitude(markers[i] - headPos);
            if (d < minD) { minD = d; sIdx = i; }
        }

        int hIdx = -1;
        float maxD = -1f;
        for (int i = 0; i < 3; i++)
        {
            if (i == sIdx) continue;
            float d = Vector3.SqrMagnitude(markers[i] - markers[sIdx]);
            if (d > maxD) { maxD = d; hIdx = i; }
        }

        int eIdx = 3 - sIdx - hIdx;

        _jointTargets[SHOULDER] = markers[sIdx];
        _jointTargets[ELBOW]    = markers[eIdx];
        _jointTargets[HAND]     = markers[hIdx];
    }

    private void MatchMarkersToJoints(List<Vector3> markers, float now)
    {
        int mc = markers.Count;
        bool[] jointTaken = new bool[3];
        bool[] markerUsed = new bool[mc];

        for (int pass = 0; pass < Mathf.Min(mc, 3); pass++)
        {
            float bestDist = float.MaxValue;
            int bestM = -1, bestJ = -1;

            for (int m = 0; m < mc; m++)
            {
                if (markerUsed[m]) continue;
                for (int j = 0; j < 3; j++)
                {
                    if (jointTaken[j]) continue;
                    float d = Vector3.SqrMagnitude(markers[m] - _jointTargets[j]);
                    if (d < bestDist) { bestDist = d; bestM = m; bestJ = j; }
                }
            }

            if (bestM >= 0)
            {
                _jointTargets[bestJ] = markers[bestM];
                _jointLastSeen[bestJ] = now;
                jointTaken[bestJ] = true;
                markerUsed[bestM] = true;
            }
        }
    }

    private Vector3 TransformPoint(Vector3 raw)
    {
        if (_synchronizer != null && _synchronizer.IsCalibrated)
            return _synchronizer.OptiTrackToUnity(raw);
        return raw;
    }

    // ================================================================
    //  Arm mode toggle
    // ================================================================

    private void ToggleArmMode()
    {
        if (!armModeActive)
        {
            bool calibrated = _synchronizer != null && _synchronizer.IsCalibrated;
            if (!calibrated)
            {
                Debug.LogWarning("[Arm] Calibrate first (C with 4 corner markers).");
                return;
            }

            armModeActive = true;
            _initialized  = false;
            _markersLost  = false;
            _lostSince    = 0f;
            _allLostTimer = 0f;
            Debug.Log("[Arm] ON — place 3 markers on shoulder, elbow, hand.");
        }
        else
        {
            armModeActive = false;
            SetVisualsActive(false);
            Debug.Log("[Arm] OFF.");
        }
    }

    // ================================================================
    //  Rigged model setup & update
    // ================================================================

    private bool SetupRiggedModel()
    {
        _modelInstance = Instantiate(armModelPrefab, transform);
        _modelInstance.name = "ArmModel_YBot";
        _modelInstance.transform.localScale = Vector3.one * modelScale;

        // Disable Animator so we drive bones manually
        var anim = _modelInstance.GetComponent<Animator>();
        if (anim != null) anim.enabled = false;

        // Search for right arm bones by name
        foreach (var t in _modelInstance.GetComponentsInChildren<Transform>())
        {
            string n = t.name;
            if (_upperArmBone == null && n.Contains("RightArm") && !n.Contains("Fore"))
                _upperArmBone = t;
            else if (_forearmBone == null && n.Contains("RightForeArm"))
                _forearmBone = t;
            else if (_handBone == null && n.Contains("RightHand")
                     && !n.Contains("Thumb") && !n.Contains("Index")
                     && !n.Contains("Middle") && !n.Contains("Ring")
                     && !n.Contains("Pinky"))
                _handBone = t;
        }

        if (_upperArmBone == null || _forearmBone == null || _handBone == null)
        {
            Debug.LogError($"[Arm] Bone search failed — UpperArm={_upperArmBone?.name}, " +
                           $"Forearm={_forearmBone?.name}, Hand={_handBone?.name}");
            Destroy(_modelInstance);
            return false;
        }

        Debug.Log($"[Arm] Bones found: {_upperArmBone.name}, {_forearmBone.name}, {_handBone.name}");

        // Record bind-pose data
        _shoulderBindOffset    = _upperArmBone.position - _modelInstance.transform.position;
        _upperArmBindLocalRot  = _upperArmBone.localRotation;
        _forearmBindLocalRot   = _forearmBone.localRotation;
        _handBindLocalPos      = _handBone.localPosition;
        _upperArmRollAxisLocal = PickRollAxis(_upperArmBone.rotation);
        _forearmRollAxisLocal  = PickRollAxis(_forearmBone.rotation);

        HideNonArmParts();

        return true;
    }

    private void HideNonArmParts()
    {
        // Reparent the upper arm bone (RightArm) directly under model root.
        // This detaches it from the spine/clavicle chain so we can collapse
        // everything else — including RightShoulder (clavicle), which was
        // causing a visible bone sticking toward the torso.
        _upperArmBone.SetParent(_modelInstance.transform, true);

        // Only keep RightArm and its children (forearm, hand, fingers)
        HashSet<Transform> keep = new HashSet<Transform>();
        AddSubtree(_upperArmBone, keep);

        // Scale every other skeleton bone to zero — collapses torso,
        // clavicle, head, legs, left arm into invisible points
        foreach (var bone in _modelInstance.GetComponentsInChildren<Transform>())
        {
            if (bone == _modelInstance.transform) continue;
            if (keep.Contains(bone)) continue;
            if (bone.GetComponent<SkinnedMeshRenderer>() != null) continue;
            if (bone.GetComponent<MeshRenderer>() != null) continue;
            bone.localScale = Vector3.zero;
        }

        foreach (var r in _modelInstance.GetComponentsInChildren<Renderer>())
            r.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;

        // Prevent SkinnedMeshRenderer from culling the arm when its static
        // bounds don't match the runtime bone positions (reparenting the
        // upper-arm bone invalidates the original bounds).
        foreach (var smr in _modelInstance.GetComponentsInChildren<SkinnedMeshRenderer>())
        {
            smr.updateWhenOffscreen = true;
            smr.forceMatrixRecalculationPerRender = true;
        }
    }

    private void AddSubtree(Transform root, HashSet<Transform> set)
    {
        set.Add(root);
        for (int i = 0; i < root.childCount; i++)
            AddSubtree(root.GetChild(i), set);
    }

    private void UpdateRiggedModel()
    {
        Vector3 shoulder = _jointSmoothed[SHOULDER];
        Vector3 elbow    = _jointSmoothed[ELBOW];
        Vector3 hand     = _jointSmoothed[HAND];

        // 1. Position model so upper-arm bone lands at the shoulder marker.
        _modelInstance.transform.position = shoulder - _shoulderBindOffset;

        // 2. Reset bones to bind pose (undo last frame's rotations + the
        //    end-of-frame hand snap, which would otherwise drift the curDir
        //    sample below frame after frame and cause the arm to spin).
        _upperArmBone.localRotation = _upperArmBindLocalRot;
        _forearmBone.localRotation  = _forearmBindLocalRot;
        _handBone.localPosition     = _handBindLocalPos;

        // 3. Rotate upper arm so it points from shoulder toward elbow.
        Vector3 curDir = (_forearmBone.position - _upperArmBone.position).normalized;
        Vector3 upperDir = (elbow - shoulder).normalized;
        if (curDir.sqrMagnitude > 0.001f && upperDir.sqrMagnitude > 0.001f)
            _upperArmBone.rotation = Quaternion.FromToRotation(curDir, upperDir) * _upperArmBone.rotation;
        StabilizeBoneRoll(_upperArmBone, upperDir, _upperArmRollAxisLocal);

        // 4. Rotate forearm so it points from elbow toward hand
        //    (forearm has moved because it's a child of upper arm).
        curDir = (_handBone.position - _forearmBone.position).normalized;
        Vector3 foreDir = (hand - elbow).normalized;
        if (curDir.sqrMagnitude > 0.001f && foreDir.sqrMagnitude > 0.001f)
            _forearmBone.rotation = Quaternion.FromToRotation(curDir, foreDir) * _forearmBone.rotation;
        StabilizeBoneRoll(_forearmBone, foreDir, _forearmRollAxisLocal);

        // The hand bone is intentionally NOT snapped to the hand marker:
        // forcing _handBone.position = hand stretches the wrist mesh past
        // what the skinning can absorb and tears it from the forearm. We
        // accept a small offset between the visible hand and the table
        // surface in exchange for a continuous arm.
    }

    // Removes the roll ambiguity left by Quaternion.FromToRotation by rotating
    // around the bone's length axis so the chosen local "up" axis projects onto
    // world up. Without this, lateral motions of the hand visibly twist it.
    private static void StabilizeBoneRoll(Transform bone, Vector3 lengthDir, Vector3 rollAxisLocal)
    {
        if (lengthDir.sqrMagnitude < 1e-4f) return;
        Vector3 currentUpWorld = bone.rotation * rollAxisLocal;
        Vector3 currentProj    = Vector3.ProjectOnPlane(currentUpWorld, lengthDir);
        Vector3 desiredProj    = Vector3.ProjectOnPlane(Vector3.up,    lengthDir);
        if (currentProj.sqrMagnitude < 1e-4f || desiredProj.sqrMagnitude < 1e-4f) return;
        float twist = Vector3.SignedAngle(currentProj.normalized, desiredProj.normalized, lengthDir);
        bone.rotation = Quaternion.AngleAxis(twist, lengthDir) * bone.rotation;
    }

    // Returns the local unit axis (±X, ±Y, ±Z) whose direction in world space
    // at bind pose was most aligned with world +Y. That axis is the natural
    // "up" reference for the bone, used by StabilizeBoneRoll.
    private static Vector3 PickRollAxis(Quaternion bindWorldRot)
    {
        Vector3[] cands = { Vector3.right, Vector3.up, Vector3.forward };
        float bestAbs = -1f;
        Vector3 best  = Vector3.up;
        for (int i = 0; i < 3; i++)
        {
            Vector3 world = bindWorldRot * cands[i];
            float dot     = Vector3.Dot(world, Vector3.up);
            if (Mathf.Abs(dot) > bestAbs)
            {
                bestAbs = Mathf.Abs(dot);
                best    = dot >= 0 ? cands[i] : -cands[i];
            }
        }
        return best;
    }

    // Body-local (right, up, gaze-forward) axis frame derived from the
    // headset orientation. Used to express the shoulder anatomical offset
    // in a frame that follows the participant rather than world space.
    private Vector3 AnatomicalShoulderOffsetWorld()
    {
        Vector3 fwd = Vector3.forward;
        if (_headset != null)
        {
            Vector3 f = Vector3.ProjectOnPlane(_headset.forward, Vector3.up);
            if (f.sqrMagnitude > 1e-4f) fwd = f.normalized;
        }
        Vector3 right = Vector3.Cross(Vector3.up, fwd);
        return right * SHOULDER_OFFSET_LOCAL.x
             + Vector3.up * SHOULDER_OFFSET_LOCAL.y
             + fwd * SHOULDER_OFFSET_LOCAL.z;
    }

    // ================================================================
    //  Fallback primitive visuals
    // ================================================================

    private void CreatePrimitives()
    {
        _primUpperArm = MakeCapsule("UpperArm", upperArmColor);
        _primForearm  = MakeCapsule("Forearm",  forearmColor);

        _primHand = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        _primHand.name = "Hand";
        _primHand.transform.SetParent(transform);
        Destroy(_primHand.GetComponent<Collider>());
        ApplyMatColor(_primHand.GetComponent<Renderer>().material, handColor);
    }

    private GameObject MakeCapsule(string name, Color color)
    {
        GameObject g = GameObject.CreatePrimitive(PrimitiveType.Capsule);
        g.name = name;
        g.transform.SetParent(transform);
        Destroy(g.GetComponent<Collider>());
        ApplyMatColor(g.GetComponent<Renderer>().material, color);
        return g;
    }

    // URP uses "_BaseColor"; legacy Standard uses "_Color" exposed as material.color.
    // Set both so the primitive arm is correctly tinted on either pipeline.
    private static void ApplyMatColor(Material mat, Color color)
    {
        if (mat.HasProperty("_BaseColor")) mat.SetColor("_BaseColor", color);
        mat.color = color;
    }

    private void UpdatePrimitives()
    {
        Vector3 shoulder = _jointSmoothed[SHOULDER];
        Vector3 elbow    = _jointSmoothed[ELBOW];
        Vector3 hand     = _jointSmoothed[HAND];

        UpdateCapsule(_primUpperArm, shoulder, elbow);
        UpdateCapsule(_primForearm,  elbow,    hand);

        _primHand.transform.position = hand;
        Vector3 dir = (hand - elbow).normalized;
        if (dir != Vector3.zero) _primHand.transform.forward = dir;
        float d = handRadius * 2f;
        _primHand.transform.localScale = new Vector3(d, d, d);
    }

    private void UpdateCapsule(GameObject capsule, Vector3 from, Vector3 to)
    {
        float length = Vector3.Distance(from, to);
        capsule.transform.position = (from + to) / 2f;
        capsule.transform.up = (to - from).normalized;
        float d = armRadius * 2f;
        capsule.transform.localScale = new Vector3(d, length / 2f, d);
    }

    // ================================================================
    //  Joint debug spheres
    // ================================================================

    private void CreateJointSpheres()
    {
        // Joint debug spheres are not created at runtime — they polluted the
        // first-person view and the visible rigged arm already conveys joint
        // positions. The fields are kept null and UpdateJointSpheres no-ops.
        if (!showJointSpheres) return;
        _shoulderSphere = MakeJointSphere("Joint_Shoulder", Color.red);
        _elbowSphere    = MakeJointSphere("Joint_Elbow",    new Color(1f, 0.7f, 0.2f));
        _handSphere     = MakeJointSphere("Joint_Hand",     Color.green);
    }

    private GameObject MakeJointSphere(string name, Color color)
    {
        GameObject g = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        g.name = name;
        g.transform.SetParent(transform);
        g.transform.localScale = Vector3.one * 0.025f;
        Destroy(g.GetComponent<Collider>());
        g.GetComponent<Renderer>().material.color = color;
        return g;
    }

    private void UpdateJointSpheres()
    {
        // Re-evaluate visibility every frame so the Inspector toggle takes
        // effect immediately. The previous version only handled the "on"
        // case, which left the spheres permanently visible after the first
        // time they had been enabled.
        bool show = showJointSpheres && armModeActive;
        if (_shoulderSphere != null) _shoulderSphere.SetActive(show);
        if (_elbowSphere    != null) _elbowSphere.SetActive(show);
        if (_handSphere     != null) _handSphere.SetActive(show);
        if (show)
        {
            _shoulderSphere.transform.position = _jointSmoothed[SHOULDER];
            _elbowSphere.transform.position    = _jointSmoothed[ELBOW];
            _handSphere.transform.position     = _jointSmoothed[HAND];
        }
    }

    // ================================================================
    //  Visibility
    // ================================================================

    private void SetVisualsActive(bool active)
    {
        if (_useModel && _modelInstance != null)
            _modelInstance.SetActive(active);

        if (!_useModel)
        {
            if (_primUpperArm != null) _primUpperArm.SetActive(active);
            if (_primForearm  != null) _primForearm.SetActive(active);
            if (_primHand     != null) _primHand.SetActive(active);
        }

        bool showJ = active && showJointSpheres;
        if (_shoulderSphere != null) _shoulderSphere.SetActive(showJ);
        if (_elbowSphere    != null) _elbowSphere.SetActive(showJ);
        if (_handSphere     != null) _handSphere.SetActive(showJ);
    }

    // ================================================================
    //  HUD
    // ================================================================

    void OnGUI()
    {
        GUIStyle style = new GUIStyle(GUI.skin.label)
        {
            fontSize  = 13,
            fontStyle = FontStyle.Bold
        };

        bool receiving  = _receiver != null && _receiver.IsReceivingData;
        bool calibrated = _synchronizer != null && _synchronizer.IsCalibrated;

        // Top-left: connection + calibration
        style.normal.textColor = receiving ? Color.green : Color.red;
        GUI.Label(new Rect(10, 10, 300, 20), receiving ? "UDP: OK" : "UDP: NO DATA", style);

        style.normal.textColor = calibrated ? Color.green : Color.yellow;
        GUI.Label(new Rect(10, 28, 300, 20), calibrated ? "CAL: OK" : "CAL: pending (C)", style);

        int count = _receiver != null ? _receiver.CurrentMarkerCount : 0;
        style.normal.textColor = Color.white;
        GUI.Label(new Rect(10, 46, 300, 20), $"Markers: {count}", style);

        string mode = _useModel ? "Model: Y Bot" : "Model: primitives";
        GUI.Label(new Rect(10, 64, 300, 20), mode, style);

        // Bottom-left: arm status
        float y = Screen.height - 30f;
        if (!armModeActive)
        {
            if (calibrated)
            {
                style.normal.textColor = Color.yellow;
                GUI.Label(new Rect(10, y, 500, 25), "Press O to activate arm mode.", style);
            }
        }
        else
        {
            string[] jn = { "Shoulder", "Elbow", "Hand" };
            float now = Time.time;

            if (visibleMarkers >= 3 && !_markersLost)
            {
                style.normal.textColor = Color.green;
                GUI.Label(new Rect(10, y, 500, 25), "ARM: 3/3 tracked", style);
            }
            else if (_initialized && _markersLost && _allLostTimer <= markerLossTimeout)
            {
                style.normal.textColor = Color.yellow;
                GUI.Label(new Rect(10, y, 600, 25),
                    $"ARM: FROZEN ({visibleMarkers}/3) — waiting for 3 markers", style);
            }
            else
            {
                style.normal.textColor = Color.red;
                GUI.Label(new Rect(10, y, 400, 25), "ARM: no markers — waiting...", style);
            }
        }
    }
}
