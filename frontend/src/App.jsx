import { useState } from 'react'
import axios from 'axios'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const FIELDS = [
  { key: 'pregnancies',      label: 'Pregnancies',            min: 0,   max: 17,  step: 1,    unit: '',        tip: 'Number of times pregnant' },
  { key: 'glucose',          label: 'Glucose',                min: 50,  max: 250, step: 1,    unit: 'mg/dL',   tip: 'Plasma glucose concentration' },
  { key: 'blood_pressure',   label: 'Blood Pressure',         min: 40,  max: 140, step: 1,    unit: 'mmHg',    tip: 'Diastolic blood pressure' },
  { key: 'skin_thickness',   label: 'Skin Thickness',         min: 0,   max: 80,  step: 1,    unit: 'mm',      tip: 'Triceps skin fold thickness' },
  { key: 'insulin',          label: 'Insulin',                min: 0,   max: 600, step: 5,    unit: 'μU/mL',   tip: '2-hour serum insulin' },
  { key: 'bmi',              label: 'BMI',                    min: 10,  max: 70,  step: 0.1,  unit: 'kg/m²',   tip: 'Body mass index' },
  { key: 'diabetes_pedigree',label: 'Diabetes Pedigree',      min: 0,   max: 2.5, step: 0.01, unit: '',        tip: 'Family history score (0–2.5)' },
  { key: 'age',              label: 'Age',                    min: 1,   max: 100, step: 1,    unit: 'yrs',     tip: 'Age in years' },
]

const DEFAULTS = {
  pregnancies: 2, glucose: 110, blood_pressure: 72, skin_thickness: 20,
  insulin: 80, bmi: 26.5, diabetes_pedigree: 0.35, age: 28,
}

const RISK_COLORS = { Low: '#10b981', Moderate: '#f59e0b', High: '#ef4444' }
const RISK_BG     = { Low: '#10b98122', Moderate: '#f59e0b22', High: '#ef444422' }

export default function App() {
  const [form, setForm]       = useState(DEFAULTS)
  const [result, setResult]   = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState('')

  const set = (key, val) => setForm(f => ({ ...f, [key]: parseFloat(val) }))

  const handleSubmit = async () => {
    setLoading(true); setError(''); setResult(null)
    try {
      const res = await axios.post(`${API}/predict`, form)
      setResult(res.data)
    } catch (e) {
      setError(e.response?.data?.detail || 'Could not connect to the prediction server.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={s.page}>
      {/* Header */}
      <div style={s.header}>
        <div style={s.pill}>ML · Healthcare</div>
        <h1 style={s.title}>Diabetes Risk Predictor</h1>
        <p style={s.subtitle}>
          Enter your health parameters below. Our Random Forest model will assess your diabetes risk
          based on the Pima Indians Diabetes Dataset.
        </p>
      </div>

      <div style={s.layout}>
        {/* Form */}
        <div style={s.card}>
          <h2 style={s.cardTitle}>Health Parameters</h2>
          <div style={s.grid}>
            {FIELDS.map(f => (
              <div key={f.key} style={s.field}>
                <div style={s.fieldHeader}>
                  <label style={s.label}>{f.label}</label>
                  <span style={s.val}>
                    {f.step < 1 ? form[f.key].toFixed(2) : form[f.key]} {f.unit}
                  </span>
                </div>
                <input
                  type="range"
                  min={f.min} max={f.max} step={f.step}
                  value={form[f.key]}
                  onChange={e => set(f.key, e.target.value)}
                  style={{ accentColor: 'var(--accent)' }}
                />
                <div style={s.tip}>{f.tip}</div>
              </div>
            ))}
          </div>

          <button style={s.btn} onClick={handleSubmit} disabled={loading}>
            {loading ? 'Analyzing...' : 'Predict Diabetes Risk'}
          </button>

          {error && <div style={s.error}>{error}</div>}
        </div>

        {/* Result */}
        <div style={s.card}>
          <h2 style={s.cardTitle}>Result</h2>

          {!result && !loading && (
            <div style={s.placeholder}>
              <div style={s.placeholderIcon}>🩺</div>
              <p style={{ color: 'var(--text2)', fontSize: '14px' }}>
                Adjust the sliders and click <strong style={{ color: 'var(--text)' }}>Predict</strong> to see your risk assessment.
              </p>
            </div>
          )}

          {loading && (
            <div style={s.placeholder}>
              <div style={s.spinner} />
              <p style={{ color: 'var(--text2)', fontSize: '14px', marginTop: '1rem' }}>Running model...</p>
            </div>
          )}

          {result && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
              {/* Risk gauge */}
              <div style={{ ...s.riskBox, background: RISK_BG[result.risk_level], borderColor: RISK_COLORS[result.risk_level] + '55' }}>
                <div style={{ fontSize: '13px', color: 'var(--text2)', marginBottom: '6px' }}>Risk Level</div>
                <div style={{ fontSize: '42px', fontWeight: 700, color: RISK_COLORS[result.risk_level] }}>
                  {result.risk_level}
                </div>
                <div style={{ fontSize: '16px', color: 'var(--text2)', marginTop: '4px' }}>
                  {result.risk_percentage}% probability
                </div>

                {/* Progress bar */}
                <div style={s.barTrack}>
                  <div style={{ ...s.barFill, width: `${result.risk_percentage}%`, background: RISK_COLORS[result.risk_level] }} />
                </div>
              </div>

              {/* Advice */}
              <div style={s.adviceBox}>
                <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text2)', marginBottom: '6px' }}>Advice</div>
                <p style={{ fontSize: '14px', lineHeight: 1.6 }}>{result.advice}</p>
              </div>

              {/* Top factors */}
              <div>
                <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text2)', marginBottom: '10px' }}>
                  Top Contributing Factors
                </div>
                {result.top_factors.map((f, i) => (
                  <div key={i} style={s.factor}>
                    <div style={s.factorHeader}>
                      <span style={{ fontSize: '13px', fontWeight: 500 }}>{f.name}</span>
                      <span style={{ fontSize: '13px', color: 'var(--text2)' }}>value: {f.value} · {f.importance}% importance</span>
                    </div>
                    <div style={s.factorTrack}>
                      <div style={{ ...s.factorFill, width: `${Math.min(f.importance * 3, 100)}%` }} />
                    </div>
                  </div>
                ))}
              </div>

              {/* Disclaimer */}
              <div style={s.disclaimer}>
                ⚠️ This tool is for educational purposes only and is not a medical diagnosis. Always consult a qualified healthcare professional.
              </div>
            </div>
          )}
        </div>
      </div>

      <div style={s.footer}>
        Built with React · FastAPI · scikit-learn · Pima Indians Diabetes Dataset
      </div>
    </div>
  )
}

const s = {
  page:          { maxWidth: '1100px', margin: '0 auto', padding: '2rem 1rem' },
  header:        { textAlign: 'center', marginBottom: '2.5rem' },
  pill:          { display: 'inline-block', background: '#3b82f622', color: '#60a5fa', fontSize: '12px', fontWeight: 600, padding: '4px 14px', borderRadius: '99px', marginBottom: '12px', border: '1px solid #3b82f644' },
  title:         { fontSize: '32px', fontWeight: 700, marginBottom: '10px' },
  subtitle:      { color: 'var(--text2)', fontSize: '15px', maxWidth: '600px', margin: '0 auto' },
  layout:        { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' },
  card:          { background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 'var(--radius-lg)', padding: '1.5rem' },
  cardTitle:     { fontSize: '16px', fontWeight: 600, marginBottom: '1.25rem', color: 'var(--text)' },
  grid:          { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.25rem', marginBottom: '1.5rem' },
  field:         { display: 'flex', flexDirection: 'column', gap: '6px' },
  fieldHeader:   { display: 'flex', justifyContent: 'space-between', alignItems: 'center' },
  label:         { fontSize: '13px', fontWeight: 500, color: 'var(--text)' },
  val:           { fontSize: '13px', color: 'var(--accent2)', fontWeight: 600 },
  tip:           { fontSize: '11px', color: 'var(--text3)' },
  btn:           { width: '100%', background: 'var(--accent)', color: '#fff', border: 'none', borderRadius: 'var(--radius)', padding: '13px', fontSize: '15px', fontWeight: 600, transition: 'opacity 0.15s' },
  error:         { marginTop: '1rem', background: '#ef444422', border: '1px solid #ef444455', borderRadius: '8px', padding: '10px 14px', fontSize: '13px', color: '#f87171' },
  placeholder:   { display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '300px', gap: '1rem', textAlign: 'center' },
  placeholderIcon: { fontSize: '48px' },
  spinner:       { width: '40px', height: '40px', border: '3px solid var(--border)', borderTop: '3px solid var(--accent)', borderRadius: '50%', animation: 'spin 0.8s linear infinite' },
  riskBox:       { border: '1px solid', borderRadius: 'var(--radius)', padding: '1.25rem', textAlign: 'center' },
  barTrack:      { height: '8px', background: 'var(--border)', borderRadius: '99px', marginTop: '12px', overflow: 'hidden' },
  barFill:       { height: '100%', borderRadius: '99px', transition: 'width 0.5s ease' },
  adviceBox:     { background: 'var(--surface2)', border: '1px solid var(--border)', borderRadius: '10px', padding: '1rem' },
  factor:        { marginBottom: '10px' },
  factorHeader:  { display: 'flex', justifyContent: 'space-between', marginBottom: '5px' },
  factorTrack:   { height: '6px', background: 'var(--border)', borderRadius: '99px', overflow: 'hidden' },
  factorFill:    { height: '100%', background: 'var(--accent)', borderRadius: '99px' },
  disclaimer:    { fontSize: '12px', color: 'var(--text3)', background: 'var(--surface2)', border: '1px solid var(--border)', borderRadius: '8px', padding: '10px 12px', lineHeight: 1.5 },
  footer:        { textAlign: 'center', marginTop: '2rem', fontSize: '13px', color: 'var(--text3)' },
}
