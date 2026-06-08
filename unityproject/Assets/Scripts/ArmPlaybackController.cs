using System;
using UnityEngine;

/// <summary>
/// Applies a real-time visuomotor perturbation to the displayed virtual hand.
/// While Play() is active, the user keeps reaching with their real arm and
/// the displayed hand is obtained by rotating the live forearm vector
/// (elbow → hand) around the live elbow pivot by an envelope-shaped angle.
/// The shoulder and elbow stay glued to their markers; only the wrist
/// position is visually displaced, so the perturbation reads as a "wrist
/// off-axis" cue rather than a whole-arm sweep. Hit detection (handled by
/// ExperimentController) operates on the real hand, so success metrics
/// reflect the participant's true motor output rather than the mismatch.
///
/// The component intentionally has no model loading or training pipeline:
/// the perturbation is fully analytical and runs at frame rate.
/// </summary>
public class ArmPlaybackController : MonoBehaviour
{
    [Header("References")]
    public ArmModelController arm;

    [Header("Perturbation")]
    [Tooltip("Time over which the envelope ramps from 0 to its full value. The angle is then held until Stop() is called.")]
    public float perturbationDuration = 1.5f;
    [Tooltip("Fraction of perturbationDuration used to ramp the envelope in (0..1). 0.4 = full strength reached at 40% of the window.")]
    [Range(0.05f, 1f)]
    public float envelopeRise = 0.4f;

    // Currently displayed angle in degrees around world-up (Y), pivoted at
    // the live real elbow position. Positive = CCW viewed from above. The
    // value is held continuously while the playback is "engaged", and
    // ramped between levels by RampTo so the visual rotation persists
    // across the four follow-up trials of a level.
    private float _currentDeg = 0f;
    private float _startDeg   = 0f;
    private float _targetDeg  = 0f;

    public float CurrentPerturbationDeg => _currentDeg;

    /// <summary>Backwards-compatible: sets the held angle without ramping.</summary>
    public void SetPerturbation(float degrees) { _currentDeg = degrees; _targetDeg = degrees; _startDeg = degrees; }

    private bool   _engaged;     // playback is taking over the displayed hand
    private bool   _ramping;     // time-based ramp (legacy)
    private float  _elapsed;
    private Action _onComplete;

    // Movement-synced ramp state. While active, the ramp progress is driven
    // by the user's spatial progress toward the target — not by a clock —
    // so the rotation builds gradually over the entire reach instead of
    // snapping into place during the first ~600 ms (which the brain easily
    // detects as a discrete event).
    private bool    _movementRamping;
    private Vector3 _movementTarget;
    private float   _movementStartDist;
    private float   _movementReachRadius;

    public bool IsPlaying => _engaged;
    public bool IsRamping => _ramping;

    void Awake()
    {
        if (arm == null) arm = FindAnyObjectByType<ArmModelController>();
    }

    /// <summary>
    /// Engages the playback override at the given starting angle (typically 0
    /// at the start of the probe phase). Marker-driven rendering is replaced
    /// by the elbow-pivoted rotation, which is then held until ReleaseHold().
    /// </summary>
    public void BeginHold(float initialDeg = 0f)
    {
        if (arm == null || !arm.IsTracking)
        {
            Debug.LogWarning("[Perturbation] Arm not tracking — cannot engage.");
            return;
        }
        _currentDeg = _startDeg = _targetDeg = initialDeg;
        _ramping    = false;
        _elapsed    = 0f;
        _onComplete = null;
        if (!_engaged) { _engaged = true; arm.BeginPlayback(); }
    }

    /// <summary>
    /// Releases the playback override and restores marker-driven rendering.
    /// Used at the end of the probe phase (or to abort everything).
    /// </summary>
    public void ReleaseHold()
    {
        if (!_engaged) return;
        _engaged = false;
        _ramping = false;
        _currentDeg = _startDeg = _targetDeg = 0f;
        arm.EndPlayback();
        Action cb = _onComplete; _onComplete = null;
        cb?.Invoke();
    }

    /// <summary>
    /// Animates the displayed perturbation from the current held angle to
    /// <paramref name="targetDeg"/> using a fixed-time SmoothStep envelope.
    /// Kept for legacy callers; new code should prefer BeginMovementRamp,
    /// which spreads the ramp across the user's entire reach and is far
    /// less noticeable.
    /// </summary>
    public void RampTo(float targetDeg, Action onComplete = null)
    {
        if (!_engaged)
        {
            Debug.LogWarning("[Perturbation] RampTo called while not engaged — engaging at 0 first.");
            BeginHold(0f);
        }
        _startDeg        = _currentDeg;
        _targetDeg       = targetDeg;
        _elapsed         = 0f;
        _ramping         = true;
        _movementRamping = false;
        _onComplete      = onComplete;
    }

    /// <summary>
    /// Starts a movement-synced ramp from the held angle to
    /// <paramref name="targetDeg"/>. The ramp progress each frame equals
    /// (smoothstep of) the fraction of the distance from <paramref name="fromHandPos"/>
    /// to <paramref name="crossPos"/> that the real hand has covered.
    /// Once the user is within <paramref name="reachRadius"/> of the cross,
    /// the angle equals <paramref name="targetDeg"/> and is held there.
    /// </summary>
    public void BeginMovementRamp(Vector3 crossPos, Vector3 fromHandPos, float targetDeg,
                                  float reachRadius, Action onComplete = null)
    {
        if (!_engaged)
        {
            Debug.LogWarning("[Perturbation] BeginMovementRamp called while not engaged — engaging at 0 first.");
            BeginHold(0f);
        }
        _startDeg            = _currentDeg;
        _targetDeg           = targetDeg;
        _movementTarget      = crossPos;
        _movementStartDist   = Vector3.Distance(fromHandPos, crossPos);
        _movementReachRadius = Mathf.Max(0.01f, reachRadius);
        _movementRamping     = true;
        _ramping             = false;
        _onComplete          = onComplete;
        Debug.Log($"[Perturbation] Movement-synced ramp {_startDeg:+0.0;-0.0}° → {_targetDeg:+0.0;-0.0}°  " +
                  $"(span {_movementStartDist*100f:F1} cm)");
    }

    /// <summary>Backwards-compatible alias kept for any callers still using Stop().</summary>
    public void Stop() => ReleaseHold();

    /// <summary>Backwards-compatible alias kept for any callers still using Play().</summary>
    public void Play(Vector3 crossPos, Action onComplete = null)
    {
        if (!_engaged) BeginHold(0f);
        RampTo(_targetDeg, onComplete);
    }

    void Update()
    {
        if (!_engaged) return;

        if (_movementRamping)
        {
            // Progress fraction along the user's reach: 0 at the start
            // position, 1 once within reachRadius of the target.
            float distNow  = Vector3.Distance(arm.RealHandPositionUnity, _movementTarget);
            float covered  = _movementStartDist - distNow;
            float span     = Mathf.Max(0.01f, _movementStartDist - _movementReachRadius);
            float progress = Mathf.Clamp01(covered / span);
            float ramp     = Mathf.SmoothStep(0f, 1f, progress);
            _currentDeg    = Mathf.LerpUnclamped(_startDeg, _targetDeg, ramp);

            if (progress >= 1f)
            {
                _currentDeg     = _targetDeg;
                _movementRamping = false;
                Action cb = _onComplete; _onComplete = null;
                cb?.Invoke();
            }
        }
        else if (_ramping)
        {
            _elapsed += Time.deltaTime;
            float tNorm = Mathf.Clamp01(_elapsed / Mathf.Max(0.05f, perturbationDuration));
            float ramp  = Mathf.SmoothStep(0f, 1f,
                            Mathf.Clamp01(tNorm / Mathf.Max(0.01f, envelopeRise)));
            _currentDeg = Mathf.LerpUnclamped(_startDeg, _targetDeg, ramp);

            if (tNorm >= 1f)
            {
                _currentDeg = _targetDeg;
                _ramping    = false;
                Action cb = _onComplete; _onComplete = null;
                cb?.Invoke();
            }
        }

        // Apply the (possibly held, possibly mid-ramp) angle by rotating the
        // live forearm vector around the live elbow pivot. Shoulder and
        // elbow stay glued to their markers; only the wrist is displaced.
        Vector3 realElbow = arm.RealElbowPositionUnity;
        Vector3 realHand  = arm.RealHandPositionUnity;
        Vector3 fromElbow = realHand - realElbow;
        Vector3 rotated   = Quaternion.AngleAxis(_currentDeg, Vector3.up) * fromElbow;
        Vector3 displayed = realElbow + rotated;

        arm.SetPlaybackHand(displayed);
    }
}
