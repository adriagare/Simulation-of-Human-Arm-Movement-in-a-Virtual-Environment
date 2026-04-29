using System;
using System.IO;
using UnityEngine;

/// <summary>
/// Loads the MLP exported by ml/train_trajectory.py (arm_model.json in
/// StreamingAssets) and drives ArmModelController autonomously to reproduce
/// the learned hand trajectory toward a given cross position.
/// </summary>
public class ArmPlaybackController : MonoBehaviour
{
    [Header("References")]
    public ArmModelController arm;

    [Header("Model")]
    public string modelFileName = "arm_model.json";

    [Header("Playback")]
    public float playbackDuration = 1.5f;

    // ── Perturbation (set by ExperimentController during the probe phase) ──
    // Rotation in degrees around the world-up axis (Y), pivoted at the
    // shoulder. Positive = counter-clockwise viewed from above. The
    // perturbation is applied to the NN-predicted hand position only.
    private float _perturbationDeg = 0f;
    private Vector3 _pivotShoulder;

    public float CurrentPerturbationDeg => _perturbationDeg;

    /// <summary>
    /// Sets the perturbation magnitude and direction in degrees applied to
    /// the NN-predicted hand around the shoulder pivot for the next Play() call.
    /// </summary>
    public void SetPerturbation(float degrees) { _perturbationDeg = degrees; }

    [Serializable]
    private class LayerJson
    {
        public int in_dim;
        public int out_dim;
        public float[] W;
        public float[] b;
        public string act;
    }

    [Serializable]
    private class ModelJson
    {
        public LayerJson[] layers;
        public float[] in_mean;
        public float[] in_std;
        public float[] out_mean;
        public float[] out_std;
    }

    private ModelJson _model;
    private bool _isPlaying;
    private float _elapsed;
    private Vector3 _crossPos;
    private Action  _onComplete;

    public bool IsPlaying => _isPlaying;
    public bool IsLoaded  => _model != null;

    void Awake()
    {
        if (arm == null) arm = FindAnyObjectByType<ArmModelController>();
        LoadModel();
    }

    public bool Reload()
    {
        _model = null;
        LoadModel();
        return _model != null;
    }

    public string ModelPath => Path.Combine(Application.streamingAssetsPath, modelFileName);

    private void LoadModel()
    {
        string path = Path.Combine(Application.streamingAssetsPath, modelFileName);
        if (!File.Exists(path))
        {
            Debug.LogWarning($"[Playback] Model not found: {path}. Train it with ml/train_trajectory.py.");
            return;
        }
        string json = File.ReadAllText(path);
        _model = JsonUtility.FromJson<ModelJson>(json);
        if (_model == null || _model.layers == null || _model.layers.Length == 0)
        {
            Debug.LogError("[Playback] Failed to parse arm_model.json");
            _model = null;
            return;
        }
        Debug.Log($"[Playback] Loaded MLP ({_model.layers.Length} layers)");
    }

    public void Play(Vector3 crossPos, Action onComplete = null)
    {
        if (arm == null || !arm.IsTracking)
        {
            Debug.LogWarning("[Playback] Arm not tracking — cannot start playback.");
            onComplete?.Invoke();
            return;
        }
        if (_model == null)
        {
            Debug.LogWarning("[Playback] No trained model — skipping playback.");
            onComplete?.Invoke();
            return;
        }
        _crossPos   = crossPos;
        _elapsed    = 0f;
        _onComplete = onComplete;
        _isPlaying  = true;
        arm.BeginPlayback();
        // Capture the shoulder pivot at playback start so the perturbation
        // rotates the predicted hand around a stable reference.
        _pivotShoulder = arm.ShoulderPositionUnity;
    }

    void Update()
    {
        if (!_isPlaying) return;

        _elapsed += Time.deltaTime;
        float tNorm = Mathf.Clamp01(_elapsed / Mathf.Max(0.05f, playbackDuration));

        Vector3 hand = Predict(_crossPos.x, _crossPos.z, tNorm);

        // Apply visuomotor perturbation: rotate the predicted hand around
        // the world-up axis, pivoted at the shoulder captured at Play().
        if (Mathf.Abs(_perturbationDeg) > 0.001f)
        {
            Vector3 fromShoulder = hand - _pivotShoulder;
            fromShoulder = Quaternion.AngleAxis(_perturbationDeg, Vector3.up) * fromShoulder;
            hand = _pivotShoulder + fromShoulder;
        }

        arm.SetPlaybackHand(hand);

        if (tNorm >= 1f)
        {
            _isPlaying = false;
            arm.EndPlayback();
            Action cb = _onComplete; _onComplete = null;
            cb?.Invoke();
        }
    }

    private Vector3 Predict(float crossX, float crossZ, float tNorm)
    {
        float[] x = new float[3];
        x[0] = (crossX - _model.in_mean[0]) / _model.in_std[0];
        x[1] = (crossZ - _model.in_mean[1]) / _model.in_std[1];
        x[2] = (tNorm  - _model.in_mean[2]) / _model.in_std[2];

        float[] a = x;
        foreach (var layer in _model.layers)
            a = Forward(layer, a);

        float hx = a[0] * _model.out_std[0] + _model.out_mean[0];
        float hy = a[1] * _model.out_std[1] + _model.out_mean[1];
        float hz = a[2] * _model.out_std[2] + _model.out_mean[2];
        return new Vector3(hx, hy, hz);
    }

    private static float[] Forward(LayerJson L, float[] inp)
    {
        float[] outp = new float[L.out_dim];
        for (int o = 0; o < L.out_dim; o++)
        {
            float s = L.b[o];
            int rowBase = o * L.in_dim;
            for (int i = 0; i < L.in_dim; i++)
                s += L.W[rowBase + i] * inp[i];
            outp[o] = L.act == "tanh" ? (float)Math.Tanh(s) : s;
        }
        return outp;
    }
}
