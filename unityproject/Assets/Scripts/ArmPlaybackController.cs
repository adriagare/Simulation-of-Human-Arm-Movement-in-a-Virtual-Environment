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

    // Rotation in degrees around the world-up axis (Y), pivoted at the
    // current real elbow position. Positive = counter-clockwise viewed
    // from above. Set by ExperimentController for each probe trial.
    private float _perturbationDeg = 0f;

    public float CurrentPerturbationDeg => _perturbationDeg;

    /// <summary>Sets the signed perturbation magnitude (deg) for the next Play().</summary>
    public void SetPerturbation(float degrees) { _perturbationDeg = degrees; }

    private bool   _isPlaying;
    private float  _elapsed;
    private Action _onComplete;

    public bool IsPlaying => _isPlaying;

    void Awake()
    {
        if (arm == null) arm = FindAnyObjectByType<ArmModelController>();
    }

    public void Play(Vector3 crossPos, Action onComplete = null)
    {
        if (arm == null || !arm.IsTracking)
        {
            Debug.LogWarning("[Perturbation] Arm not tracking — cannot start.");
            onComplete?.Invoke();
            return;
        }
        _elapsed    = 0f;
        _onComplete = onComplete;
        _isPlaying  = true;
        arm.BeginPlayback();
        // Pivot is sampled live each frame inside Update — the elbow moves
        // appreciably during the reach (shoulder fixed, forearm sweeping),
        // so a frozen pivot would drift the perturbation away from the
        // real wrist over the course of a trial.
    }

    /// <summary>
    /// Aborts an in-flight perturbation (e.g. when the user reaches the
    /// cross before the envelope finishes). Restores marker-driven
    /// rendering and fires the completion callback so the controller can
    /// advance phase.
    /// </summary>
    public void Stop()
    {
        if (!_isPlaying) return;
        _isPlaying = false;
        arm.EndPlayback();
        Action cb = _onComplete; _onComplete = null;
        cb?.Invoke();
    }

    void Update()
    {
        if (!_isPlaying) return;

        _elapsed += Time.deltaTime;
        float tNorm = Mathf.Clamp01(_elapsed / Mathf.Max(0.05f, perturbationDuration));

        // Smoothly ramp the perturbation in; once at full strength it is
        // held until Stop() is invoked by the experiment controller.
        float ramp = Mathf.SmoothStep(0f, 1f,
                       Mathf.Clamp01(tNorm / Mathf.Max(0.01f, envelopeRise)));
        float effectiveDeg = ramp * _perturbationDeg;

        // Rotate the live forearm vector around the live elbow pivot. The
        // shoulder and elbow stay anchored to their markers; only the
        // wrist position is visually displaced.
        Vector3 realElbow = arm.RealElbowPositionUnity;
        Vector3 realHand  = arm.RealHandPositionUnity;
        Vector3 fromElbow = realHand - realElbow;
        Vector3 rotated   = Quaternion.AngleAxis(effectiveDeg, Vector3.up) * fromElbow;
        Vector3 displayed = realElbow + rotated;

        arm.SetPlaybackHand(displayed);

        if (tNorm >= 1f && _onComplete != null)
        {
            // Notify once when the envelope completes; perturbation keeps
            // running at full strength until Stop() is called.
            Action cb = _onComplete; _onComplete = null;
            cb.Invoke();
        }
    }
}
