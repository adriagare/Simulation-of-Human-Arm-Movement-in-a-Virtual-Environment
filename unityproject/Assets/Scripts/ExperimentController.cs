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

    [Header("Playback Phase")]
    public int   playbackTrials       = 5;
    public float playbackRestDelay    = 1.5f;  // wait after elbow returns to rest zone
    public float playbackPostDelay    = 1.0f;  // pause after each autonomous movement

    [Header("Probe Phase (visuomotor detection threshold)")]
    [Tooltip("If true, after fine-tune the experiment runs the perturbation-detection probe instead of the silent autonomous playback.")]
    public bool   enableProbePhase    = false;
    [Tooltip("Magnitudes (deg) of the rotation applied to the NN-predicted hand around the shoulder. A 0 entry serves as control.")]
    public float[] probeAngles        = new float[] { 0f, 5f, 10f, 15f, 20f, 25f, 30f, 45f };
    [Tooltip("Number of trials per perturbation level.")]
    public int    trialsPerProbeLevel = 5;
    [Tooltip("Randomize the sign (±) of the perturbation each probe trial. Recommended on to avoid motor adaptation.")]
    public bool   randomizeProbeDirection = true;
    [Tooltip("Randomize the order of probe levels. If off, levels are presented from smallest to largest.")]
    public bool   randomizeProbeOrder     = false;

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

    [Header("Fine-tuning (pretrain + finetune pipeline)")]
    [Tooltip("If true, after data collection launches Python to fine-tune the NN on this session's CSV before playback.")]
    public bool   autoFinetune    = true;
    [Tooltip("Absolute or PATH-resolvable Python executable. e.g. python, python3, C:/Python311/python.exe")]
    public string pythonExecutable = "python";
    [Tooltip("Script path relative to the project root (folder containing Assets/).")]
    public string trainScriptPath = "../ml/train_trajectory.py";
    public float  finetuneTimeout = 120f; // seconds
    [Tooltip("Trial index (1-based) at which background fine-tune is launched in parallel with remaining trials. 0 = disabled (wait until end).")]
    public int    backgroundFinetuneTriggerTrial = 25;

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
        WaitingForTraining,
        PlaybackWaitRest, PlaybackPreCue, PlaybackActive, PlaybackPost, PlaybackDone,
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

    private int   _playbackIndex  = 0;
    private float _playbackHold   = 0f;
    private System.Diagnostics.Process _trainProc;
    private float _trainTimer     = 0f;
    private bool  _trainLaunched  = false;
    private string _snapshotCsvPath;

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

        if (_phase == Phase.Idle || _phase == Phase.PlaybackDone || _phase == Phase.ProbeDone) return;

        // While Python is fine-tuning, skip logging (CSV already closed).
        if (_phase == Phase.WaitingForTraining) { TickFinetune(); return; }

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
                    MaybeLaunchBackgroundFinetune();
                    if (_trialIndex >= _trialTarget) EndExperiment();
                    else _phase = Phase.WaitingForRest;
                }
                break;

            // ── Autonomous NN playback phase ────────────────────────
            case Phase.PlaybackWaitRest:
                if (HandInRestZone())
                {
                    _playbackHold -= Time.deltaTime;
                    if (_playbackHold <= 0f) BeginPlaybackTrial();
                }
                else _playbackHold = playbackRestDelay;
                break;

            case Phase.PlaybackPreCue:
                _phaseTimer -= Time.deltaTime;
                if (_phaseTimer <= 0f) StartAutonomousMovement();
                break;

            case Phase.PlaybackActive:
                // Driven by ArmPlaybackController; completion callback advances phase.
                break;

            case Phase.PlaybackPost:
                _phaseTimer -= Time.deltaTime;
                if (_phaseTimer <= 0f)
                {
                    _playbackIndex++;
                    if (_playbackIndex >= playbackTrials) EndPlayback();
                    else _phase = Phase.PlaybackWaitRest;
                }
                break;

            // ── Probe (perturbation-detection) phase ────────────────────
            case Phase.ProbeWaitRest:
                if (HandInRestZone())
                {
                    _playbackHold -= Time.deltaTime;
                    if (_playbackHold <= 0f) BeginProbeTrial();
                }
                else _playbackHold = playbackRestDelay;
                break;

            case Phase.ProbePreCue:
                _phaseTimer -= Time.deltaTime;
                if (_phaseTimer <= 0f) StartProbeMovement();
                break;

            case Phase.ProbeActive:
                // Driven by ArmPlaybackController; completion callback advances phase.
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
        _trainLaunched  = false;
        _snapshotCsvPath = null;
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
        CloseCSV();

        // If background fine-tune was launched mid-experiment, just wait for it.
        if (_trainLaunched && _trainProc != null)
        {
            _phase = Phase.WaitingForTraining;
            return;
        }

        if (autoFinetune)
        {
            if (LaunchFinetune(_csvPath))
            {
                _phase      = Phase.WaitingForTraining;
                _trainTimer = 0f;
                _trainLaunched = true;
                return;
            }
            Debug.LogWarning("[Exp] Fine-tune launch failed — skipping.");
        }
        EnterAutonomousPhase();
    }

    private void MaybeLaunchBackgroundFinetune()
    {
        if (!autoFinetune) return;
        if (_trainLaunched) return;
        if (backgroundFinetuneTriggerTrial <= 0) return;
        if (_trialIndex < backgroundFinetuneTriggerTrial) return;
        if (_csv == null || _csvPath == null) return;

        // Flush pending rows so the snapshot has all completed trials.
        try { _csv.Flush(); } catch { }

        // Copy to a snapshot so Python can read while Unity keeps appending
        // (avoids Windows file-sharing violations).
        _snapshotCsvPath = _csvPath.Replace(".csv", "_snapshot.csv");
        try
        {
            File.Copy(_csvPath, _snapshotCsvPath, overwrite: true);
        }
        catch (System.Exception e)
        {
            Debug.LogWarning($"[Exp] Snapshot copy failed: {e.Message}");
            return;
        }

        if (LaunchFinetune(_snapshotCsvPath))
        {
            _trainLaunched = true;
            _trainTimer    = 0f;
            Debug.Log($"[Exp] Background fine-tune started at trial {_trialIndex} (snapshot: {Path.GetFileName(_snapshotCsvPath)})");
        }
    }

    private void EnterAutonomousPhase()
    {
        // Branch between the legacy silent-playback flow and the new probe
        // (perturbation-detection) flow. The probe flow keeps fine-tune,
        // pretrain, etc. — only the post-data-collection phase changes.
        if (enableProbePhase) EnterProbePhase();
        else                  EnterPlaybackPhase();
    }

    private void EnterPlaybackPhase()
    {
        if (playback == null || !playback.IsLoaded)
        {
            Debug.LogWarning("[Exp] No playback model loaded — ending without NN phase.");
            _phase = Phase.PlaybackDone;
            DestroyRestZone();
            return;
        }
        ReopenCSVAppend();   // resume logging so user reaction is recorded
        _playbackIndex = 0;
        _playbackHold  = playbackRestDelay;
        _phase = Phase.PlaybackWaitRest;
    }

    // ────────────────────────────────────────────────────────────────
    //  Probe (perturbation-detection) phase
    // ────────────────────────────────────────────────────────────────

    private void EnterProbePhase()
    {
        if (playback == null || !playback.IsLoaded)
        {
            Debug.LogWarning("[Exp/Probe] No playback model loaded — cannot run probe phase.");
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

        ReopenCSVAppend();

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
        _playbackHold      = playbackRestDelay;

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
        _phase       = Phase.ProbePreCue;
        _phaseTimer  = preCueDelay;

        Debug.Log($"[Exp/Probe] Level {_probeOrderIdx + 1}/{_probeOrder.Count} " +
                  $"(θ = {_currentProbeAngle:+0.0;-0.0}°)  " +
                  $"trial {_probeTrialInLevel + 1}/{trialsPerProbeLevel}");
    }

    private void StartProbeMovement()
    {
        _phase = Phase.ProbeActive;
        playback.Play(_crossPos, OnProbeMovementDone);
    }

    private void OnProbeMovementDone()
    {
        DestroyCross();
        _probeTrialInLevel++;
        _phase      = Phase.ProbePost;
        _phaseTimer = playbackPostDelay;
        Debug.Log($"[Exp/Probe] Trial done — θ={_currentProbeAngle:+0.0;-0.0}°  " +
                  $"({_probeTrialInLevel}/{trialsPerProbeLevel} at this level)");
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
        _phase        = Phase.ProbeWaitRest;
        _playbackHold = playbackRestDelay;
    }

    private void EndProbe()
    {
        Debug.Log("[Exp/Probe] Probe phase finished.");
        playback?.SetPerturbation(0f);
        DestroyRestZone();
        _phase = Phase.ProbeDone;
    }

    private void ReopenCSVAppend()
    {
        if (_csv != null || _csvPath == null) return;
        try { _csv = new StreamWriter(_csvPath, true); }
        catch (System.Exception e) { Debug.LogWarning($"[Exp] Could not reopen CSV: {e.Message}"); }
    }

    private bool LaunchFinetune(string csvPath)
    {
        try
        {
            string projectRoot = Path.GetFullPath(Path.Combine(Application.dataPath, ".."));
            string script = Path.GetFullPath(Path.Combine(projectRoot, trainScriptPath));
            if (!File.Exists(script))
            {
                Debug.LogError($"[Exp] Train script not found: {script}");
                return false;
            }
            if (csvPath == null || !File.Exists(csvPath))
            {
                Debug.LogError($"[Exp] CSV missing: {csvPath}");
                return false;
            }

            var psi = new System.Diagnostics.ProcessStartInfo
            {
                FileName         = pythonExecutable,
                Arguments        = $"\"{script}\" --finetune \"{csvPath}\"",
                WorkingDirectory = Path.GetDirectoryName(script),
                UseShellExecute  = false,
                CreateNoWindow   = true,
                RedirectStandardOutput = true,
                RedirectStandardError  = true,
            };
            _trainProc = System.Diagnostics.Process.Start(psi);
            Debug.Log($"[Exp] Fine-tune launched: {pythonExecutable} {psi.Arguments}");
            return _trainProc != null;
        }
        catch (System.Exception e)
        {
            Debug.LogError($"[Exp] Launch error: {e.Message}");
            return false;
        }
    }

    private void TickFinetune()
    {
        _trainTimer += Time.deltaTime;

        if (_trainProc != null && _trainProc.HasExited)
        {
            int code = _trainProc.ExitCode;
            string stdout = _trainProc.StandardOutput.ReadToEnd();
            string stderr = _trainProc.StandardError.ReadToEnd();
            _trainProc.Dispose();
            _trainProc = null;

            if (code != 0)
            {
                Debug.LogError($"[Exp] Fine-tune failed (exit {code}).\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}");
                EnterAutonomousPhase();   // fall back to whatever JSON exists
                return;
            }
            Debug.Log("[Exp] Fine-tune OK. Reloading model.\n" + stdout);

            if (playback != null) playback.Reload();
            if (_snapshotCsvPath != null && File.Exists(_snapshotCsvPath))
            {
                try { File.Delete(_snapshotCsvPath); } catch { }
            }
            EnterAutonomousPhase();
            return;
        }

        if (_trainTimer > finetuneTimeout)
        {
            Debug.LogError("[Exp] Fine-tune timeout — proceeding with existing model.");
            try { _trainProc?.Kill(); } catch { }
            _trainProc?.Dispose();
            _trainProc = null;
            EnterAutonomousPhase();
        }
    }

    private void BeginPlaybackTrial()
    {
        _crossPos    = RandomCrossPosition();
        _crossGO     = SpawnCross(_crossPos);
        _silentTrial = false;
        _trialStart  = Time.time;
        _phase       = Phase.PlaybackPreCue;
        _phaseTimer  = preCueDelay;
        Debug.Log($"[Exp/NN] Playback {_playbackIndex + 1}/{playbackTrials}  pos=({_crossPos.x:F2},{_crossPos.z:F2})");
    }

    private void StartAutonomousMovement()
    {
        // SILENT autonomous movement — no cue sound. We want to observe how
        // the user reacts to the arm moving when no stimulus told them to.
        _phase = Phase.PlaybackActive;
        playback.Play(_crossPos, OnAutonomousMovementDone);
    }

    private void OnAutonomousMovementDone()
    {
        DestroyCross();
        _phase      = Phase.PlaybackPost;
        _phaseTimer = playbackPostDelay;
    }

    private void EndPlayback()
    {
        Debug.Log("[Exp/NN] Autonomous playback finished.");
        _phase = Phase.PlaybackDone;
        DestroyRestZone();
        CloseCSV();
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
            "{21:F4},{22:F4},{23:F4},{24:F4}," +
            "{25},{26},{27:F2},{28}",
            Time.time - _trialStart, _trialIndex, _phase,
            _silentTrial ? 1 : 0, _trialScored ? 1 : 0, _score,
            _crossPos.x, _crossPos.y, _crossPos.z,
            handU.x, handU.y, handU.z,
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
            case Phase.WaitingForTraining:
                s.normal.textColor = Color.cyan;
                GUI.Label(new Rect(10, y, 800, 20),
                    $"Personalising NN model...  ({_trainTimer:F0}s)", s);
                break;
            case Phase.PlaybackWaitRest:
                s.normal.textColor = Color.magenta;
                GUI.Label(new Rect(10, y, 800, 20),
                    $"NN {_playbackIndex + 1}/{playbackTrials}  —  place elbow in GREEN zone and keep still", s);
                break;
            case Phase.PlaybackPreCue:
            case Phase.PlaybackActive:
                s.normal.textColor = Color.magenta;
                GUI.Label(new Rect(10, y, 800, 20),
                    $"NN {_playbackIndex + 1}/{playbackTrials}  —  arm moves autonomously, DO NOT move", s);
                break;
            case Phase.PlaybackPost:
                s.normal.textColor = Color.magenta;
                GUI.Label(new Rect(10, y, 800, 20),
                    $"NN trial {_playbackIndex + 1}/{playbackTrials} done", s);
                break;
            case Phase.PlaybackDone:
                s.normal.textColor = Color.green;
                GUI.Label(new Rect(10, y, 800, 20), "Session complete.", s);
                break;

            // ── Probe phase ─────────────────────────────────────────
            case Phase.ProbeWaitRest:
                s.normal.textColor = Color.yellow;
                GUI.Label(new Rect(10, y, 800, 20),
                    $"PROBE  level {_probeOrderIdx + 1}/{(_probeOrder != null ? _probeOrder.Count : 0)}  " +
                    $"trial {_probeTrialInLevel + 1}/{trialsPerProbeLevel}  —  return hand to rest zone", s);
                break;
            case Phase.ProbePreCue:
            case Phase.ProbeActive:
                s.normal.textColor = new Color(1f, 0.6f, 1f);
                GUI.Label(new Rect(10, y, 900, 20),
                    $"PROBE  θ = {_currentProbeAngle:+0.0;-0.0}°  " +
                    $"level {_probeOrderIdx + 1}/{(_probeOrder != null ? _probeOrder.Count : 0)}  " +
                    $"trial {_probeTrialInLevel + 1}/{trialsPerProbeLevel}", s);
                break;
            case Phase.ProbePost:
                s.normal.textColor = Color.magenta;
                GUI.Label(new Rect(10, y, 900, 20),
                    $"PROBE  θ={_currentProbeAngle:+0.0;-0.0}°  trial done   [M=mark  1-4=labels]", s);
                break;
            case Phase.ProbeDone:
                s.normal.textColor = Color.green;
                GUI.Label(new Rect(10, y, 800, 20), "Probe phase complete.", s);
                break;
        }
    }
}
