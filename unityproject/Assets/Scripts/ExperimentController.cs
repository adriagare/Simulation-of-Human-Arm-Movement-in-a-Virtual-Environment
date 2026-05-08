using System.Collections.Generic;
using System.Globalization;
using System.IO;
using UnityEngine;
using UnityEngine.InputSystem;

/// <summary>
/// Neuroscience experiment module — Approach B (interleaved).
///
/// Flow:
///   1. Press B after arm mode is active (O). Score starts at 0.
///   2. Hand must be inside the green REST ZONE to arm the next trial.
///   3. Trial picks type randomly (sound vs silent) by <see cref="silentProbability"/>.
///      - SOUND trial:
///          * Spawn red cross, wait preCueDelay (1s), play start.mp3.
///          * User has hitTimeout (2s) to move hand within detectionRadius.
///              hit  → +1 point, itempicker.mp3
///              miss → 0 points, wrong.mp3
///      - SILENT trial:
///          * Spawn red cross, wait preCueDelay, DO NOT play start.mp3.
///          * Observe hand for silentWindow (2s):
///              stays within stillnessThreshold of spawn position → +1 point
///              moves more than threshold                         → 0 points, wrong.mp3
///   4. After each trial the user must return hand to rest zone to start next.
///   5. After <random(minTrials, maxTrials)> trials, experiment ends.
///
/// Per-frame data is logged to Positions/experiment_&lt;timestamp&gt;.csv.
/// </summary>
public class ExperimentController : MonoBehaviour
{
    // ────────────────────────────────────────────────────────────────
    //  Inspector
    // ────────────────────────────────────────────────────────────────

    [Header("References")]
    public ArmModelController   arm;
    public UDPMarkerReceiver    receiver;
    public ArmPlaybackController playback;

    [Header("Probe Phase (visuomotor detection threshold)")]
    [Tooltip("Magnitudes (deg) of the perturbation rotation applied to the displayed hand. A 0 entry serves as control.")]
    public float[] probeAngles        = new float[] { 0f, 5f, 10f, 15f, 20f, 25f, 30f, 45f };
    [Tooltip("Number of trials per perturbation level.")]
    public int    trialsPerProbeLevel = 5;
    [Tooltip("Randomize the sign (±) of the perturbation each probe trial. Recommended on to avoid motor adaptation.")]
    public bool   randomizeProbeDirection = true;
    [Tooltip("Randomize the order of probe levels. If off, levels are presented from smallest to largest.")]
    public bool   randomizeProbeOrder     = false;
    [Tooltip("Pause between the elbow returning to the rest zone and the next probe trial starting.")]
    public float  probeRestDelay      = 1.5f;
    [Tooltip("Pause after each probe trial completes before advancing to the next.")]
    public float  probePostDelay      = 1.0f;

    [Header("Spontaneous-event annotations")]
    [Tooltip("Generic mark: experimenter presses this when something noteworthy happens (label = 'mark').")]
    public Key    annotationGenericKey = Key.M;
    [Tooltip("Predefined label 1: participant verbalised that the arm felt strange.")]
    public Key    annotationKey1 = Key.Digit1;
    public string annotationLabel1 = "user_said_strange";
    [Tooltip("Predefined label 2: participant looked at the virtual arm with visible surprise.")]
    public Key    annotationKey2 = Key.Digit2;
    public string annotationLabel2 = "user_visibly_surprised";
    [Tooltip("Predefined label 3: participant stopped or hesitated mid-trial.")]
    public Key    annotationKey3 = Key.Digit3;
    public string annotationLabel3 = "user_hesitated";
    [Tooltip("Predefined label 4: any other reaction worth marking.")]
    public Key    annotationKey4 = Key.Digit4;
    public string annotationLabel4 = "other";

    [Header("Audio Clips")]
    public AudioClip startClip;        // cue to move  (sound trial)
    public AudioClip itempickerClip;   // successful hit / correct still
    public AudioClip wrongClip;        // miss / moved during silent trial

    [Header("Experiment Parameters")]
    public Key   activateKey       = Key.B;
    public int   minTrials         = 40;
    public int   maxTrials         = 60;
    public float preCueDelay       = 1.0f;   // cross visible → start.mp3 (or silent begin)
    public float hitTimeout        = 2.0f;   // sound: time allowed to reach cross
    public float silentWindow      = 2.0f;   // silent: observation window
    public float postTrialDelay    = 1.0f;   // brief pause showing result
    public float detectionRadius   = 0.10f;  // hand-to-cross contact (10 cm)
    public float crossSize         = 0.09f;
    [Range(0f, 1f)]
    public float silentProbability = 0.30f;  // chance a trial is silent
    public float stillnessThreshold = 0.05f; // 5 cm allowed hand drift during silent trial
    public float firstTrialDelay    = 1.5f;  // grace period after elbow enters rest zone before first cross
    public float minCrossDistFromRest = 0.25f; // min XZ distance between cross and rest-zone center

    [Header("Cross Spawn Area")]
    [Tooltip("Cross is spawned at random X ∈ [xMin,xMax], Z ∈ [zMin,zMax], Y = tableSurfaceY. Bounds chosen so the user can reach any cross without leaning forward (shoulder must stay still — the NN playback assumes a frozen shoulder).")]
    public float xMin = -0.25f;
    public float xMax =  0.45f;
    public float zMin =  0.55f;   // keep crosses away from the rest zone
    public float zMax =  0.75f;   // arm-only reach (no torso lean)
    public float tableSurfaceY = 1.155f;

    [Header("Rest Zone (green rectangle, closest to user)")]
    public Vector3 restZoneCenter = new Vector3(0f, 1.155f, 0.42f);
    public Vector2 restZoneSize   = new Vector2(0.30f, 0.12f);  // X, Z extents
    public float   restYTolerance = 0.15f;                        // hand-height tolerance

    [Header("Output")]
    public string logFolder = "Positions";

    // ────────────────────────────────────────────────────────────────
    //  State
    // ────────────────────────────────────────────────────────────────

    private enum Phase {
        Idle, WaitingForRest, PreCue, ActiveTrial, PostTrial, Done,
        ProbeWaitRest, ProbePreCue, ProbeActive, ProbePost, ProbeDone
    }

    // ── Probe phase state ─────────────────────────────────────────────
    private List<int> _probeOrder;          // indices into probeAngles, in presentation order
    private int   _probeOrderIdx     = 0;   // current entry in _probeOrder
    private int   _probeTrialInLevel = 0;   // 1..trialsPerProbeLevel within current level
    private float _currentProbeAngle = 0f;  // signed angle applied this trial (deg)
    private int   _currentProbeDir   = +1;  // ±1

    // Side-channel events log: timestamp + phase + trial + angle + label.
    private StreamWriter _eventCsv;
    private string       _eventCsvPath;

    // Countdown timer used between probe trials while waiting for the elbow
    // to settle back into the rest zone. Reset on every trial start.
    private float _probeRestHold   = 0f;

    private Phase   _phase       = Phase.Idle;
    private int     _trialIndex  = -1;
    private int     _trialTarget = 0;
    private int     _score       = 0;
    private float   _phaseTimer  = 0f;
    private float   _trialStart  = 0f;
    private bool    _silentTrial = false;
    private bool    _trialScored = false;
    private Vector3 _handAtSpawn;      // hand position when cross appeared (silent trial / precue baseline)
    private float   _maxDrift    = 0f; // max deviation during silent window
    private Vector3 _handAtPreCue;     // hand position at start of PreCue (sound trial early-move check)
    private float   _firstTrialHold  = 0f; // countdown while elbow is in rest zone before first trial

    private GameObject  _crossGO;
    private Vector3     _crossPos;
    private GameObject  _restZoneGO;
    private AudioSource _audio;
    private Transform   _headset;


    private Vector3 _lastHandPos;
    private float   _lastHandTime;
    private Vector3 _handVelocity;

    private StreamWriter _csv;
    private string       _csvPath;

    // ────────────────────────────────────────────────────────────────
    //  Unity lifecycle
    // ────────────────────────────────────────────────────────────────

    void Awake()
    {
        if (arm == null)      arm      = FindAnyObjectByType<ArmModelController>();
        if (receiver == null) receiver = FindAnyObjectByType<UDPMarkerReceiver>();
        if (playback == null) playback = FindAnyObjectByType<ArmPlaybackController>();

        _audio = gameObject.AddComponent<AudioSource>();
        _audio.spatialBlend = 0f;
        _audio.playOnAwake  = false;

        var cam = Camera.main;
        if (cam != null) _headset = cam.transform;
    }

    void Update()
    {
        if (_phase == Phase.Idle
            && Keyboard.current != null
            && Keyboard.current[activateKey].wasPressedThisFrame)
        {
            if (arm == null || !arm.IsArmModeActive)
            {
                Debug.LogWarning("[Exp] Activate arm mode (O) before starting experiment (B).");
                return;
            }
            BeginExperiment();
            return;
        }

        if (_phase == Phase.Idle || _phase == Phase.ProbeDone) return;

        UpdateHandKinematics();
        LogFrame();
        TickAnnotations();

        switch (_phase)
        {
            case Phase.WaitingForRest:
                if (HandInRestZone())
                {
                    if (_trialIndex == 0)
                    {
                        _firstTrialHold -= Time.deltaTime;
                        if (_firstTrialHold <= 0f) StartNextTrial();
                    }
                    else
                    {
                        StartNextTrial();
                    }
                }
                else
                {
                    _firstTrialHold = firstTrialDelay;  // reset if elbow leaves
                }
                break;

            case Phase.PreCue:
                _phaseTimer -= Time.deltaTime;
                // Early-move penalty: if user moves hand before the cue (or before silent-window start),
                // it counts as a mistake. Applies to both sound and silent trials.
                if (arm != null && arm.IsTracking
                    && Vector3.Distance(arm.HandPositionUnity, _handAtPreCue) > stillnessThreshold)
                {
                    Play(wrongClip);
                    FinishTrial(false);
                    break;
                }
                if (_phaseTimer <= 0f) BeginActivePhase();
                break;

            case Phase.ActiveTrial:
                _phaseTimer -= Time.deltaTime;
                if (_silentTrial)
                {
                    TrackSilentDrift();
                    if (_phaseTimer <= 0f) EndSilentTrial();
                }
                else
                {
                    if (HandInsideCross())      { EndSoundTrial(true);  }
                    else if (_phaseTimer <= 0f) { EndSoundTrial(false); }
                }
                break;

            case Phase.PostTrial:
                _phaseTimer -= Time.deltaTime;
                if (_phaseTimer <= 0f)
                {
                    _trialIndex++;
                    if (_trialIndex >= _trialTarget) EndExperiment();
                    else _phase = Phase.WaitingForRest;
                }
                break;

            // ── Probe (perturbation-detection) phase ────────────────────
            case Phase.ProbeWaitRest:
                if (HandInRestZone())
                {
                    _probeRestHold -= Time.deltaTime;
                    if (_probeRestHold <= 0f) BeginProbeTrial();
                }
                else _probeRestHold = probeRestDelay;
                break;

            case Phase.ProbePreCue:
                _phaseTimer -= Time.deltaTime;
                // Early-move penalty (real hand): user must keep still until cue.
                if (arm != null && arm.IsTracking
                    && Vector3.Distance(arm.RealHandPositionUnity, _handAtPreCue) > stillnessThreshold)
                {
                    EndProbeTrial(false);
                    break;
                }
                if (_phaseTimer <= 0f) StartProbeMovement();
                break;

            case Phase.ProbeActive:
                _phaseTimer -= Time.deltaTime;
                // Hit detection uses the REAL hand so success isn't biased by
                // the visual perturbation. Trial ends on hit or hitTimeout.
                if (RealHandInsideCross())   { EndProbeTrial(true); }
                else if (_phaseTimer <= 0f)  { EndProbeTrial(false); }
                break;

            case Phase.ProbePost:
                _phaseTimer -= Time.deltaTime;
                if (_phaseTimer <= 0f) AdvanceProbeOrLoop();
                break;
        }
    }

    void OnDestroy() { CloseCSV(); DestroyRestZone(); }

    // ────────────────────────────────────────────────────────────────
    //  Experiment flow
    // ────────────────────────────────────────────────────────────────

    private void BeginExperiment()
    {
        _trialTarget = Random.Range(minTrials, maxTrials + 1);
        _trialIndex  = 0;
        _score       = 0;
        OpenCSV();
        SpawnRestZone();
        _firstTrialHold = firstTrialDelay;
        Debug.Log($"[Exp] Started. Target trials: {_trialTarget}. Log: {_csvPath}");
        _phase = Phase.WaitingForRest;
    }

    private void StartNextTrial()
    {
        _silentTrial = Random.value < silentProbability;
        _crossPos    = RandomCrossPosition();
        _crossGO     = SpawnCross(_crossPos);
        _trialStart  = Time.time;
        _trialScored = false;
        _maxDrift    = 0f;
        _handAtPreCue = (arm != null && arm.IsTracking) ? arm.HandPositionUnity : Vector3.zero;
        _phase       = Phase.PreCue;
        _phaseTimer  = preCueDelay;

        Debug.Log($"[Exp] Trial {_trialIndex + 1}/{_trialTarget}  " +
                  $"pos=({_crossPos.x:F2},{_crossPos.y:F2},{_crossPos.z:F2})  " +
                  $"silent={_silentTrial}");
    }

    private void BeginActivePhase()
    {
        if (_silentTrial)
        {
            _handAtSpawn = arm.IsTracking ? arm.HandPositionUnity : Vector3.zero;
            _phaseTimer  = silentWindow;
        }
        else
        {
            Play(startClip);
            _phaseTimer = hitTimeout;
        }
        _phase = Phase.ActiveTrial;
    }

    private void EndSoundTrial(bool hit)
    {
        if (hit) { Play(itempickerClip); _score++; }
        else     { Play(wrongClip); }
        FinishTrial(hit);
    }

    private void EndSilentTrial()
    {
        bool stayedStill = _maxDrift <= stillnessThreshold;
        if (stayedStill) { Play(itempickerClip); _score++; }
        else             { Play(wrongClip); }
        FinishTrial(stayedStill);
    }

    private void FinishTrial(bool success)
    {
        _trialScored = true;
        DestroyCross();
        _phase      = Phase.PostTrial;
        _phaseTimer = postTrialDelay;
    }

    private void EndExperiment()
    {
        Debug.Log("[Exp] Data collection complete.");
        EnterProbePhase();
    }

    // ────────────────────────────────────────────────────────────────
    //  Probe (perturbation-detection) phase
    // ────────────────────────────────────────────────────────────────

    private void EnterProbePhase()
    {
        if (playback == null)
        {
            Debug.LogWarning("[Exp/Probe] No perturbation controller wired — cannot run probe phase.");
            _phase = Phase.ProbeDone;
            DestroyRestZone();
            return;
        }
        if (probeAngles == null || probeAngles.Length == 0)
        {
            Debug.LogWarning("[Exp/Probe] probeAngles is empty — skipping probe phase.");
            _phase = Phase.ProbeDone;
            DestroyRestZone();
            return;
        }

        // Build presentation order
        _probeOrder = new List<int>(probeAngles.Length);
        for (int i = 0; i < probeAngles.Length; i++) _probeOrder.Add(i);
        if (randomizeProbeOrder)
        {
            for (int i = _probeOrder.Count - 1; i > 0; i--)
            {
                int j = Random.Range(0, i + 1);
                (_probeOrder[i], _probeOrder[j]) = (_probeOrder[j], _probeOrder[i]);
            }
        }
        _probeOrderIdx     = 0;
        _probeTrialInLevel = 0;
        _phase             = Phase.ProbeWaitRest;
        _probeRestHold     = probeRestDelay;

        Debug.Log($"[Exp/Probe] Starting probe phase: {probeAngles.Length} levels × {trialsPerProbeLevel} trials.");
    }

    private void BeginProbeTrial()
    {
        // Pick the angle and direction for this trial
        int angleIdx = _probeOrder[_probeOrderIdx];
        float magnitude = Mathf.Abs(probeAngles[angleIdx]);
        _currentProbeDir   = randomizeProbeDirection ? (Random.value < 0.5f ? -1 : +1) : +1;
        _currentProbeAngle = magnitude * _currentProbeDir;

        playback.SetPerturbation(_currentProbeAngle);

        _crossPos    = RandomCrossPosition();
        _crossGO     = SpawnCross(_crossPos);
        _silentTrial = false;
        _trialStart  = Time.time;
        _trialScored = false;
        _handAtPreCue = (arm != null && arm.IsTracking) ? arm.RealHandPositionUnity : Vector3.zero;
        _phase       = Phase.ProbePreCue;
        _phaseTimer  = preCueDelay;

        Debug.Log($"[Exp/Probe] Level {_probeOrderIdx + 1}/{_probeOrder.Count} " +
                  $"(θ = {_currentProbeAngle:+0.0;-0.0}°)  " +
                  $"trial {_probeTrialInLevel + 1}/{trialsPerProbeLevel}");
    }

    private void StartProbeMovement()
    {
        _phase      = Phase.ProbeActive;
        _phaseTimer = hitTimeout;
        Play(startClip);
        // Playback shapes the visual mismatch on top of the user's real hand.
        // Its completion callback only fires if the envelope finishes before
        // the user reaches the cross or the hit timeout elapses.
        playback.Play(_crossPos, OnProbeMovementDone);
    }

    private void OnProbeMovementDone()
    {
        // Called when the perturbation envelope finishes naturally (i.e. the
        // user neither reached the cross nor timed out before playbackDuration).
        // The trial keeps running until ProbeActive resolves via hit/timeout.
    }

    private void EndProbeTrial(bool hit)
    {
        if (playback != null && playback.IsPlaying) playback.Stop();
        if (hit) { Play(itempickerClip); _score++; }
        else     { Play(wrongClip); }
        _trialScored = true;
        DestroyCross();
        _probeTrialInLevel++;
        _phase      = Phase.ProbePost;
        _phaseTimer = probePostDelay;
        Debug.Log($"[Exp/Probe] Trial done — θ={_currentProbeAngle:+0.0;-0.0}°  hit={hit}  " +
                  $"({_probeTrialInLevel}/{trialsPerProbeLevel} at this level)");
    }

    private bool RealHandInsideCross()
    {
        if (arm == null || !arm.IsTracking) return false;
        return Vector3.Distance(arm.RealHandPositionUnity, _crossPos) < detectionRadius;
    }

    private void AdvanceProbeOrLoop()
    {
        if (_probeTrialInLevel >= trialsPerProbeLevel)
        {
            _probeOrderIdx++;
            _probeTrialInLevel = 0;
            if (_probeOrderIdx >= _probeOrder.Count)
            {
                EndProbe();
                return;
            }
        }
        _phase         = Phase.ProbeWaitRest;
        _probeRestHold = probeRestDelay;
    }

    private void EndProbe()
    {
        Debug.Log("[Exp/Probe] Probe phase finished.");
        playback?.SetPerturbation(0f);
        DestroyRestZone();
        CloseCSV();
        _phase = Phase.ProbeDone;
    }

    // ────────────────────────────────────────────────────────────────
    //  Silent-trial drift tracking
    // ────────────────────────────────────────────────────────────────

    private void TrackSilentDrift()
    {
        if (arm == null || !arm.IsTracking) return;
        float d = Vector3.Distance(arm.HandPositionUnity, _handAtSpawn);
        if (d > _maxDrift) _maxDrift = d;
    }

    // ────────────────────────────────────────────────────────────────
    //  Rest zone
    // ────────────────────────────────────────────────────────────────

    private void SpawnRestZone()
    {
        if (_restZoneGO != null) return;
        _restZoneGO = GameObject.CreatePrimitive(PrimitiveType.Cube);
        _restZoneGO.name = "ExperimentRestZone";
        Destroy(_restZoneGO.GetComponent<Collider>());
        _restZoneGO.transform.position   = restZoneCenter;
        _restZoneGO.transform.localScale = new Vector3(restZoneSize.x, 0.004f, restZoneSize.y);

        Shader sh = Shader.Find("Universal Render Pipeline/Unlit");
        if (sh == null) sh = Shader.Find("Unlit/Color");
        Material mat = new Material(sh);
        Color green = new Color(0.2f, 0.85f, 0.3f, 1f);
        if (mat.HasProperty("_BaseColor")) mat.SetColor("_BaseColor", green);
        mat.color = green;
        _restZoneGO.GetComponent<Renderer>().material = mat;
    }

    private void DestroyRestZone()
    {
        if (_restZoneGO != null) Destroy(_restZoneGO);
        _restZoneGO = null;
    }

    private bool HandInRestZone()
    {
        // Rest zone is judged by the ELBOW marker — the elbow is the stable pivot
        // that actually returns to the rest pose, while the hand can drift slightly.
        if (arm == null || !arm.IsTracking) return false;
        Vector3 e = arm.ElbowPositionUnity;
        Vector3 c = restZoneCenter;
        return Mathf.Abs(e.x - c.x) <= restZoneSize.x * 0.5f
            && Mathf.Abs(e.z - c.z) <= restZoneSize.y * 0.5f
            && Mathf.Abs(e.y - c.y) <= restYTolerance;
    }

    // ────────────────────────────────────────────────────────────────
    //  Cross spawn & detection
    // ────────────────────────────────────────────────────────────────

    private Vector3 RandomCrossPosition()
    {
        for (int i = 0; i < 20; i++)
        {
            Vector3 p = new Vector3(
                Random.Range(xMin, xMax),
                tableSurfaceY,
                Random.Range(zMin, zMax));
            float dx = p.x - restZoneCenter.x;
            float dz = p.z - restZoneCenter.z;
            if (Mathf.Sqrt(dx * dx + dz * dz) >= minCrossDistFromRest) return p;
        }
        // fallback: push Z forward to guarantee distance
        return new Vector3(
            Random.Range(xMin, xMax),
            tableSurfaceY,
            Mathf.Max(zMin, restZoneCenter.z + minCrossDistFromRest));
    }

    private GameObject SpawnCross(Vector3 position)
    {
        GameObject cross = new GameObject("ExperimentCross");
        cross.transform.position = position;

        Shader sh = Shader.Find("Universal Render Pipeline/Unlit");
        if (sh == null) sh = Shader.Find("Unlit/Color");
        Material mat = new Material(sh);
        if (mat.HasProperty("_BaseColor")) mat.SetColor("_BaseColor", Color.red);
        mat.color = Color.red;

        float bar   = crossSize;
        float thick = crossSize * 0.18f;

        GameObject h = GameObject.CreatePrimitive(PrimitiveType.Cube);
        h.name = "bar_h";
        h.transform.SetParent(cross.transform, false);
        h.transform.localScale = new Vector3(bar, thick, thick);
        Destroy(h.GetComponent<Collider>());
        h.GetComponent<Renderer>().material = mat;

        GameObject v = GameObject.CreatePrimitive(PrimitiveType.Cube);
        v.name = "bar_v";
        v.transform.SetParent(cross.transform, false);
        v.transform.localScale = new Vector3(thick, thick, bar);
        Destroy(v.GetComponent<Collider>());
        v.GetComponent<Renderer>().material = mat;

        return cross;
    }

    private void DestroyCross()
    {
        if (_crossGO != null) Destroy(_crossGO);
        _crossGO = null;
    }

    private bool HandInsideCross()
    {
        if (arm == null || !arm.IsTracking) return false;
        return Vector3.Distance(arm.HandPositionUnity, _crossPos) < detectionRadius;
    }

    // ────────────────────────────────────────────────────────────────
    //  Kinematics
    // ────────────────────────────────────────────────────────────────

    private void UpdateHandKinematics()
    {
        if (arm == null || !arm.IsTracking) return;
        Vector3 p  = arm.HandPositionUnity;
        float   t  = Time.time;
        float   dt = Mathf.Max(t - _lastHandTime, 1e-4f);
        Vector3 v  = (p - _lastHandPos) / dt;
        _handVelocity = Vector3.Lerp(_handVelocity, v, 0.5f);
        _lastHandPos  = p;
        _lastHandTime = t;
    }

    // ────────────────────────────────────────────────────────────────
    //  Audio
    // ────────────────────────────────────────────────────────────────

    private void Play(AudioClip c)
    {
        if (c == null) return;
        _audio.PlayOneShot(c);
    }

    // ────────────────────────────────────────────────────────────────
    //  CSV logging
    // ────────────────────────────────────────────────────────────────

    private void OpenCSV()
    {
        string dir = Path.Combine(Application.dataPath, "..", logFolder);
        if (!Directory.Exists(dir)) Directory.CreateDirectory(dir);
        string stamp = System.DateTime.Now.ToString("yyyyMMdd_HHmmss");
        _csvPath = Path.Combine(dir, $"experiment_{stamp}.csv");
        _csv = new StreamWriter(_csvPath, false);
        _csv.WriteLine(
            "timestamp,trial,phase,silent,scored,score," +
            "cross_x,cross_y,cross_z," +
            "hand_unity_x,hand_unity_y,hand_unity_z," +
            "hand_real_x,hand_real_y,hand_real_z," +
            "elbow_real_x,elbow_real_y,elbow_real_z," +
            "shoulder_real_x,shoulder_real_y,shoulder_real_z," +
            "hand_raw_x,hand_raw_y,hand_raw_z," +
            "vel_x,vel_y,vel_z," +
            "head_pos_x,head_pos_y,head_pos_z," +
            "head_rot_x,head_rot_y,head_rot_z,head_rot_w," +
            "probe_level,probe_trial,probe_angle,probe_dir");
        _csv.Flush();
    }

    private void LogFrame()
    {
        if (_csv == null || arm == null) return;
        Vector3 handU = arm.IsTracking ? arm.HandPositionUnity : Vector3.zero;
        // *_real_* columns track the marker-derived joints even while the
        // displayed arm is overridden by the perturbation module. The elbow
        // is the live pivot of the visuomotor rotation, so logging it lets
        // the analyst reconstruct the geometry offline. The shoulder is
        // included for full upper-body kinematics.
        Vector3 handReal     = arm.IsTracking ? arm.RealHandPositionUnity     : Vector3.zero;
        Vector3 elbowReal    = arm.IsTracking ? arm.RealElbowPositionUnity    : Vector3.zero;
        Vector3 shoulderReal = arm.IsTracking ? arm.RealShoulderPositionUnity : Vector3.zero;
        Vector3 handR = GetHandRaw();
        Vector3 hp    = _headset != null ? _headset.position : Vector3.zero;
        Quaternion hr = _headset != null ? _headset.rotation : Quaternion.identity;
        var ci = CultureInfo.InvariantCulture;
        // Probe-phase columns: only meaningful while in a Probe* state.
        bool inProbe = _phase == Phase.ProbeWaitRest || _phase == Phase.ProbePreCue
                    || _phase == Phase.ProbeActive   || _phase == Phase.ProbePost;
        int probeLevel    = inProbe ? _probeOrderIdx + 1 : 0;
        int probeTrial    = inProbe ? _probeTrialInLevel + 1 : 0;
        float probeAngle  = inProbe ? _currentProbeAngle : 0f;
        int probeDir      = inProbe ? _currentProbeDir   : 0;

        _csv.WriteLine(string.Format(ci,
            "{0:F4},{1},{2},{3},{4},{5}," +
            "{6:F4},{7:F4},{8:F4}," +
            "{9:F4},{10:F4},{11:F4}," +
            "{12:F4},{13:F4},{14:F4}," +
            "{15:F4},{16:F4},{17:F4}," +
            "{18:F4},{19:F4},{20:F4}," +
            "{21:F4},{22:F4},{23:F4}," +
            "{24:F4},{25:F4},{26:F4}," +
            "{27:F4},{28:F4},{29:F4}," +
            "{30:F4},{31:F4},{32:F4},{33:F4}," +
            "{34},{35},{36:F2},{37}",
            Time.time - _trialStart, _trialIndex, _phase,
            _silentTrial ? 1 : 0, _trialScored ? 1 : 0, _score,
            _crossPos.x, _crossPos.y, _crossPos.z,
            handU.x, handU.y, handU.z,
            handReal.x, handReal.y, handReal.z,
            elbowReal.x, elbowReal.y, elbowReal.z,
            shoulderReal.x, shoulderReal.y, shoulderReal.z,
            handR.x, handR.y, handR.z,
            _handVelocity.x, _handVelocity.y, _handVelocity.z,
            hp.x, hp.y, hp.z,
            hr.x, hr.y, hr.z, hr.w,
            probeLevel, probeTrial, probeAngle, probeDir));
    }

    private Vector3 GetHandRaw()
    {
        if (receiver == null) return Vector3.zero;
        Dictionary<int, Vector3> raws = receiver.GetAllRawPositions();
        foreach (var kv in raws) return kv.Value;
        return Vector3.zero;
    }

    private void CloseCSV()
    {
        if (_csv != null) { _csv.Flush(); _csv.Close(); _csv = null; }
        if (_eventCsv != null) { _eventCsv.Flush(); _eventCsv.Close(); _eventCsv = null; }
    }

    // ────────────────────────────────────────────────────────────────
    //  Spontaneous-event annotations
    // ────────────────────────────────────────────────────────────────

    private void OpenEventCsv()
    {
        if (_eventCsv != null) return;
        try
        {
            string dir = Path.Combine(Application.dataPath, "..", logFolder);
            if (!Directory.Exists(dir)) Directory.CreateDirectory(dir);
            string stamp = System.DateTime.Now.ToString("yyyyMMdd_HHmmss");
            _eventCsvPath = Path.Combine(dir, $"events_{stamp}.csv");
            _eventCsv = new StreamWriter(_eventCsvPath, false);
            _eventCsv.WriteLine("time,trial,phase,probe_level,probe_angle,label");
            _eventCsv.Flush();
        }
        catch (System.Exception e)
        {
            Debug.LogWarning($"[Exp] Could not open events CSV: {e.Message}");
            _eventCsv = null;
        }
    }

    private void WriteEvent(string label)
    {
        if (_eventCsv == null) OpenEventCsv();
        if (_eventCsv == null) return;
        var ci = CultureInfo.InvariantCulture;
        bool inProbe = _phase == Phase.ProbeWaitRest || _phase == Phase.ProbePreCue
                    || _phase == Phase.ProbeActive   || _phase == Phase.ProbePost;
        int probeLevel    = inProbe ? _probeOrderIdx + 1 : 0;
        float probeAngle  = inProbe ? _currentProbeAngle : 0f;
        _eventCsv.WriteLine(string.Format(ci,
            "{0:F4},{1},{2},{3},{4:F2},{5}",
            Time.time - _trialStart, _trialIndex, _phase,
            probeLevel, probeAngle, label));
        _eventCsv.Flush();
        Debug.Log($"[Exp/Event] {label}  (trial {_trialIndex}, θ={probeAngle:+0.0;-0.0}°)");
    }

    private void TickAnnotations()
    {
        if (Keyboard.current == null) return;

        if (Keyboard.current[annotationGenericKey].wasPressedThisFrame) WriteEvent("mark");
        if (Keyboard.current[annotationKey1].wasPressedThisFrame)       WriteEvent(annotationLabel1);
        if (Keyboard.current[annotationKey2].wasPressedThisFrame)       WriteEvent(annotationLabel2);
        if (Keyboard.current[annotationKey3].wasPressedThisFrame)       WriteEvent(annotationLabel3);
        if (Keyboard.current[annotationKey4].wasPressedThisFrame)       WriteEvent(annotationLabel4);
    }

    // ────────────────────────────────────────────────────────────────
    //  HUD
    // ────────────────────────────────────────────────────────────────

    void OnGUI()
    {
        GUIStyle s = new GUIStyle(GUI.skin.label) { fontSize = 13, fontStyle = FontStyle.Bold };
        s.normal.textColor = Color.cyan;
        float y = Screen.height - 55f;

        switch (_phase)
        {
            case Phase.Idle:
                if (arm != null && arm.IsArmModeActive)
                    GUI.Label(new Rect(10, y, 700, 20), "Press B to start experiment.", s);
                break;
            case Phase.WaitingForRest:
                s.normal.textColor = Color.yellow;
                GUI.Label(new Rect(10, y, 700, 20),
                    $"EXP {_trialIndex + 1}/{_trialTarget}  —  place hand in GREEN zone", s);
                break;
            case Phase.PreCue:
            case Phase.ActiveTrial:
                string tag = _silentTrial ? "SILENT" : "SOUND";
                GUI.Label(new Rect(10, y, 700, 20),
                    $"EXP {_trialIndex + 1}/{_trialTarget}  [{tag}]  phase={_phase}", s);
                break;
            case Phase.PostTrial:
                GUI.Label(new Rect(10, y, 700, 20),
                    $"Trial {_trialIndex + 1}/{_trialTarget} complete", s);
                break;
            case Phase.Done:
                s.normal.textColor = Color.green;
                GUI.Label(new Rect(10, y, 700, 20),
                    $"Experiment complete.", s);
                break;
            // ── Probe phase ─────────────────────────────────────────
            case Phase.ProbeWaitRest:
                s.normal.textColor = Color.yellow;
                GUI.Label(new Rect(10, y, 800, 20),
                    $"PROBE  level {_probeOrderIdx + 1}/{(_probeOrder != null ? _probeOrder.Count : 0)}  " +
                    $"trial {_probeTrialInLevel + 1}/{trialsPerProbeLevel}  —  return hand to rest zone", s);
                break;
            case Phase.ProbePreCue:
                s.normal.textColor = new Color(1f, 0.6f, 1f);
                GUI.Label(new Rect(10, y, 900, 20),
                    $"PROBE  level {_probeOrderIdx + 1}/{(_probeOrder != null ? _probeOrder.Count : 0)}  " +
                    $"trial {_probeTrialInLevel + 1}/{trialsPerProbeLevel}  —  wait for cue", s);
                break;
            case Phase.ProbeActive:
                s.normal.textColor = new Color(1f, 0.6f, 1f);
                GUI.Label(new Rect(10, y, 900, 20),
                    $"PROBE  level {_probeOrderIdx + 1}/{(_probeOrder != null ? _probeOrder.Count : 0)}  " +
                    $"trial {_probeTrialInLevel + 1}/{trialsPerProbeLevel}  —  REACH for the cross", s);
                break;
            case Phase.ProbePost:
                s.normal.textColor = Color.magenta;
                GUI.Label(new Rect(10, y, 900, 20),
                    $"PROBE  trial done   [M=mark  1-4=labels]", s);
                break;
            case Phase.ProbeDone:
                s.normal.textColor = Color.green;
                GUI.Label(new Rect(10, y, 800, 20), "Probe phase complete.", s);
                break;
        }
    }
}
