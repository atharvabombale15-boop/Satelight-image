@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700&family=Space+Grotesk:wght@400;600&display=swap');

:root {
  --bg-1: #04070c;
  --bg-2: #071521;
  --panel: rgba(10, 24, 36, 0.75);
  --panel-solid: #0a1a28;
  --glow: #3af3ff;
  --accent: #7ef9ff;
  --text: #eef7ff;
  --muted: #9eb4c8;
  --line: rgba(58, 243, 255, 0.28);
  --shadow: 0 22px 70px rgba(0, 0, 0, 0.5);
}

.app {
  min-height: 100vh;
  padding: 40px clamp(16px, 4vw, 48px) 60px;
  background: radial-gradient(1000px 600px at 15% -10%, #14253a 0%, transparent 60%),
    radial-gradient(900px 520px at 90% 0%, #103249 0%, transparent 55%),
    linear-gradient(180deg, var(--bg-1) 0%, var(--bg-2) 100%);
  color: var(--text);
  position: relative;
  overflow-x: hidden;
}

.app::before {
  content: "";
  position: fixed;
  inset: 0;
  pointer-events: none;
  background: repeating-linear-gradient(
    180deg,
    rgba(58, 243, 255, 0.05) 0px,
    rgba(58, 243, 255, 0.05) 1px,
    rgba(0, 0, 0, 0) 3px,
    rgba(0, 0, 0, 0) 6px
  );
  opacity: 0.22;
  mix-blend-mode: screen;
  animation: scan 8s linear infinite;
  z-index: 0;
}

.app > * {
  position: relative;
  z-index: 1;
}

.hero {
  display: flex;
  gap: 24px;
  justify-content: space-between;
  align-items: flex-start;
  padding: 28px;
  border-radius: 22px;
  border: 1px solid var(--line);
  background: linear-gradient(135deg, rgba(58, 243, 255, 0.12), rgba(126, 249, 255, 0.08));
  box-shadow: var(--shadow);
  animation: rise 700ms ease-out, pulse 6s ease-in-out infinite;
}

.hero h1 {
  font-family: 'Orbitron', sans-serif;
  letter-spacing: 0.08em;
  margin: 0 0 10px;
}

.hero .subtitle {
  color: var(--muted);
  margin: 0;
  max-width: 520px;
}

.hero-pills {
  display: flex;
  flex-direction: column;
  gap: 10px;
  font-size: 0.85rem;
}

.hero-pills span {
  padding: 6px 12px;
  border-radius: 999px;
  border: 1px solid var(--line);
  background: rgba(8, 18, 26, 0.6);
  color: var(--muted);
}

.grid {
  display: grid;
  grid-template-columns: minmax(0, 2fr) minmax(0, 1fr);
  gap: 24px;
  margin-top: 24px;
}

.panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 18px;
  padding: 22px;
  box-shadow: var(--shadow);
  backdrop-filter: blur(10px);
  margin-top: 20px;
}

.panel h2 {
  font-family: 'Orbitron', sans-serif;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  font-size: 0.95rem;
  color: var(--glow);
  margin-bottom: 16px;
}

.upload-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 16px;
}

.upload-card {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 16px;
  border-radius: 14px;
  border: 1px dashed rgba(58, 243, 255, 0.5);
  background: var(--panel-solid);
  cursor: pointer;
}

.upload-card input {
  margin-top: 6px;
}

.upload-card .label {
  font-weight: 600;
}

.upload-card .hint {
  color: var(--muted);
  font-size: 0.9rem;
}

.status-card {
  padding: 16px;
  border-radius: 14px;
  border: 1px solid var(--line);
  background: rgba(8, 18, 26, 0.7);
}

.status-card h3 {
  margin: 0 0 8px;
  font-family: 'Orbitron', sans-serif;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  font-size: 0.85rem;
  color: var(--glow);
}

.run-button {
  margin-top: 12px;
  padding: 10px 16px;
  border-radius: 999px;
  border: 1px solid rgba(58, 243, 255, 0.4);
  background: rgba(10, 26, 40, 0.7);
  color: var(--text);
  cursor: pointer;
  font-weight: 600;
  transition: transform 200ms ease, box-shadow 200ms ease;
}

.run-button:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.run-button:not(:disabled):hover {
  transform: translateY(-2px);
  box-shadow: 0 14px 30px rgba(58, 243, 255, 0.2);
}

.error {
  color: #ff8f8f;
  margin-top: 10px;
}

.settings .setting {
  display: flex;
  flex-direction: column;
  gap: 10px;
  margin-bottom: 16px;
}

.settings input[type="range"] {
  accent-color: var(--glow);
}

.settings select {
  padding: 8px;
  border-radius: 8px;
  border: 1px solid var(--line);
  background: rgba(8, 18, 26, 0.8);
  color: var(--text);
}

.settings input[type="number"] {
  padding: 8px;
  border-radius: 8px;
  border: 1px solid var(--line);
  background: rgba(8, 18, 26, 0.8);
  color: var(--text);
}

.meta {
  margin-top: 12px;
  display: grid;
  gap: 6px;
}

.quality {
  display: grid;
  grid-template-columns: 140px 1fr 60px;
  align-items: center;
  gap: 10px;
  color: var(--muted);
  font-size: 0.9rem;
}

.quality-bar {
  height: 8px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.08);
  overflow: hidden;
}

.quality-fill {
  height: 100%;
  background: linear-gradient(90deg, rgba(58, 243, 255, 0.4), rgba(126, 249, 255, 0.9));
  box-shadow: 0 0 12px rgba(58, 243, 255, 0.6);
}

.settings .checkbox label {
  display: flex;
  gap: 10px;
  align-items: center;
}

.preview-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 16px;
}

.preview-card {
  display: grid;
  gap: 8px;
  padding: 12px;
  border-radius: 14px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  background: rgba(8, 18, 26, 0.6);
  text-align: center;
}

.preview-card img {
  width: 100%;
  border-radius: 12px;
  box-shadow: 0 18px 40px rgba(0, 0, 0, 0.45);
  max-height: 320px;
  object-fit: cover;
}

.slider-block {
  display: grid;
  gap: 12px;
}

.slider-wrap {
  position: relative;
  width: min(640px, 100%);
  aspect-ratio: 16 / 10;
  border-radius: 16px;
  overflow: hidden;
  border: 1px solid rgba(58, 243, 255, 0.35);
  box-shadow: 0 22px 60px rgba(0, 0, 0, 0.55);
}

.slider-wrap img {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.slider-wrap .tag {
  position: absolute;
  top: 12px;
  padding: 6px 10px;
  font-size: 0.8rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  border-radius: 999px;
  border: 1px solid rgba(58, 243, 255, 0.4);
  background: rgba(8, 18, 26, 0.65);
  color: #eef7ff;
  z-index: 3;
}

.slider-wrap .tag.left {
  left: 12px;
}

.slider-wrap .tag.right {
  right: 12px;
}

.metric-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 16px;
}

.metric-card {
  padding: 14px;
  border-radius: 16px;
  border: 1px solid rgba(58, 243, 255, 0.35);
  background: linear-gradient(160deg, rgba(58, 243, 255, 0.18), rgba(126, 249, 255, 0.08));
  box-shadow: 0 18px 50px rgba(0, 0, 0, 0.4);
  animation: pulse 6s ease-in-out infinite;
}

.metric-card h3 {
  font-family: 'Orbitron', sans-serif;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  font-size: 0.8rem;
  margin: 0 0 8px;
  color: var(--glow);
}

.metric-card p {
  font-size: 1.6rem;
  margin: 0;
}

.download-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 16px;
}

.download-button {
  padding: 12px 16px;
  border-radius: 999px;
  border: 1px solid rgba(58, 243, 255, 0.4);
  background: rgba(10, 26, 40, 0.7);
  color: var(--text);
  text-align: center;
  text-decoration: none;
  font-weight: 600;
  transition: transform 200ms ease, box-shadow 200ms ease;
}

.download-button:hover {
  transform: translateY(-2px);
  box-shadow: 0 14px 30px rgba(58, 243, 255, 0.2);
}

.batch-row {
  display: flex;
  gap: 12px;
  align-items: center;
  flex-wrap: wrap;
}

.batch-results {
  margin-top: 12px;
  display: grid;
  gap: 10px;
}

.mission-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 16px;
}

.mission-card {
  padding: 16px;
  border-radius: 14px;
  border: 1px solid var(--line);
  background: rgba(8, 18, 26, 0.6);
}

.mission-card h3 {
  font-family: 'Orbitron', sans-serif;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  font-size: 0.8rem;
  margin: 0 0 10px;
  color: var(--glow);
}

.footer {
  margin-top: 28px;
  padding-top: 14px;
  border-top: 1px solid rgba(255, 255, 255, 0.08);
  color: var(--muted);
  font-size: 0.85rem;
}

.muted {
  color: var(--muted);
}

.legend {
  color: var(--muted);
  font-size: 0.85rem;
  margin: 0 0 12px;
}

@keyframes scan {
  0% {
    transform: translateY(-10%);
  }
  100% {
    transform: translateY(10%);
  }
}

@keyframes rise {
  0% {
    opacity: 0;
    transform: translateY(16px);
  }
  100% {
    opacity: 1;
    transform: translateY(0);
  }
}

@keyframes pulse {
  0% {
    box-shadow: 0 18px 50px rgba(0, 0, 0, 0.4);
    border-color: rgba(58, 243, 255, 0.35);
  }
  50% {
    box-shadow: 0 22px 70px rgba(58, 243, 255, 0.18);
    border-color: rgba(58, 243, 255, 0.6);
  }
  100% {
    box-shadow: 0 18px 50px rgba(0, 0, 0, 0.4);
    border-color: rgba(58, 243, 255, 0.35);
  }
}

@media (max-width: 980px) {
  .hero {
    flex-direction: column;
  }

  .grid {
    grid-template-columns: 1fr;
  }

  .preview-card img {
    max-height: 220px;
  }

  .slider-wrap {
    width: min(520px, 100%);
    aspect-ratio: 4 / 3;
  }
}
