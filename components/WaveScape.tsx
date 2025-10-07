// FIX: Add `useCallback` to the import from 'react' to resolve 'Cannot find name' errors.
import React, { useEffect, useMemo, useRef, useState, useCallback } from "react";

// WaveScape Gradient Descent — single-file React component
// Drop this file into your Next.js app (e.g., app/components/WaveScape.tsx)
// and render <WaveScape/> anywhere. No external deps. Tailwind optional.

// ------------------------------
// Math — loss landscape
// ------------------------------

// Parameters to define the shape of the loss landscape
interface Gaussian { amp: number; x: number; y: number; sx: number; sy: number; }
interface Sinusoid { amp: number; freq_x: number; freq_y: number; phase_x: number; phase_y: number; }
interface LandscapeParams {
  gaussians: Gaussian[];
  sinusoids: Sinusoid[];
  base_quadratic: { xx: number; yy: number; xy: number; };
}

const rand = (min: number, max: number) => min + Math.random() * (max - min);

// Generates a new set of random parameters for the landscape
function generateRandomParams(): LandscapeParams {
    const gaussians: Gaussian[] = [];
    const numGaussians = 6;
    for (let i = 0; i < numGaussians; i++) {
        gaussians.push({
            amp: rand(-2.5, 2.5), // Increased amplitude for more pronounced hills/valleys
            x: rand(-4, 4),
            y: rand(-4, 4),
            sx: rand(0.5, 1.5),
            sy: rand(0.5, 1.5)
        });
    }

    const sinusoids: Sinusoid[] = [];
    const numSinusoids = 2;
    for (let i = 0; i < numSinusoids; i++) {
        sinusoids.push({
            amp: rand(0.5, 1.2), // Increased amplitude for more pronounced waves
            freq_x: rand(0.5, 1.5),
            freq_y: rand(0.5, 1.5),
            phase_x: rand(0, Math.PI * 2),
            phase_y: rand(0, Math.PI * 2)
        });
    }

    return {
        gaussians,
        sinusoids,
        base_quadratic: {
            xx: rand(0.01, 0.04),
            yy: rand(0.01, 0.04),
            xy: rand(-0.02, 0.02)
        }
    };
}


// ------------------------------
// Learning-rate schedules
// ------------------------------
const SCHEDULES = ["constant", "linear", "cosine", "cosine-restarts"] as const;
export type ScheduleKind = (typeof SCHEDULES)[number];

function lrSchedule(
  kind: ScheduleKind,
  base: number,
  step: number,
  stepsPerCycle: number
): number {
  const t = Math.max(0, Math.min(1, step / stepsPerCycle));
  switch (kind) {
    case "linear":
      // Linear decay to ~0
      return base * (1 - t);
    case "cosine":
      return base * 0.5 * (1 + Math.cos(Math.PI * t));
    case "cosine-restarts": {
      // Cosine annealing with warm restarts: restart at each cycle
      const localT = (step % stepsPerCycle) / stepsPerCycle;
      return base * 0.5 * (1 + Math.cos(Math.PI * localT));
    }
    case "constant":
    default:
      return base;
  }
}

// ------------------------------
// Palette helper — Map height to color
// ------------------------------
function heightToColor(z: number, minZ: number, maxZ: number, alpha = 0.9): string {
    const range = maxZ - minZ;
    const t = range > 1e-6 ? (z - minZ) / range : 0.5; // normalized to [0, 1]
    // HSL: hue from purple (deep) -> blue -> cyan -> green -> yellow -> red (high)
    const hue = 260 - t * 280;
    const lightness = 40 + t * 30;
    const saturation = 90 - t * 15;
    return `hsla(${hue}, ${saturation}%, ${lightness}%, ${alpha})`;
}

// Box-Muller transform to get a standard normal random variable
function randomNormal() {
    let u1 = 0, u2 = 0;
    while(u1 === 0) u1 = Math.random(); // Converting [0,1) to (0,1)
    while(u2 === 0) u2 = Math.random();
    return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
}


// ------------------------------
// Component
// ------------------------------
export const WaveScape: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const rafRef = useRef<number | null>(null);
  const [isRunning, setIsRunning] = useState(false); // Start paused on reset
  const [landscapeParams, setLandscapeParams] = useState(generateRandomParams);

  // Simulation parameters
  const [optimizer, setOptimizer] = useState<'sgd' | 'adam'>('adam');
  const [baseLR, setBaseLR] = useState(0.03);
  const [momentum, setMomentum] = useState(0.85); // For SGD
  const [nesterov, setNesterov] = useState(true); // For SGD
  const [schedule, setSchedule] = useState<ScheduleKind>("cosine-restarts");
  const [cycleSeconds, setCycleSeconds] = useState(4);

  // Advanced techniques
  const [temperature, setTemperature] = useState(0.2); // Langevin Dynamics
  const [annealTemp, setAnnealTemp] = useState(true); // Temp annealing
  const [blur, setBlur] = useState(1.0); // Continuation method

  // Landscape drawing details
  const settings = useMemo(
    () => ({
      worldMin: -5,
      worldMax: 5,
      nLines: 46, // number of ridge-lines
      samplesPerLine: 200,
      heightScale: 80, // vertical exaggeration in px per world unit z
      lineWidth: 1.2,
    }),
    []
  );

  // Dynamic loss function based on current landscape parameters and blur
  const lossFn = useCallback((x: number, y: number, currentBlur: number): number => {
    const p = landscapeParams;
    let z = 0;

    // Base quadratic bowl
    const q = p.base_quadratic;
    z += q.xx * x * x + q.yy * y * y + q.xy * x * y;

    const blurFactor = 1.0 - currentBlur;
    // Sinusoidal waves
    for (const s of p.sinusoids) {
        z += blurFactor * s.amp * Math.cos(s.freq_x * x + s.phase_x) * Math.sin(s.freq_y * y + s.phase_y);
    }
    
    // Gaussian peaks and valleys
    for (const g of p.gaussians) {
        const dx = x - g.x;
        const dy = y - g.y;
        z += blurFactor * g.amp * Math.exp(-(dx*dx / (2 * g.sx * g.sx) + dy*dy / (2 * g.sy * g.sy)));
    }

    return z;
  }, [landscapeParams]);

  const gradLoss = useCallback((x: number, y: number, currentBlur: number): { gx: number; gy: number } => {
    const h = 1e-4;
    const gx = (lossFn(x + h, y, currentBlur) - lossFn(x - h, y, currentBlur)) / (2 * h);
    const gy = (lossFn(x, y + h, currentBlur) - lossFn(x, y - h, currentBlur)) / (2 * h);
    return { gx, gy };
  }, [lossFn]);

  // Find landscape Z-bounds for color mapping
  const { minLoss, maxLoss } = useMemo(() => {
    let minL = Infinity, maxL = -Infinity;
    const { worldMin, worldMax } = settings;
    const samples = 80;
    for (let i = 0; i < samples; i++) {
        for (let j = 0; j < samples; j++) {
            const x = worldMin + (i / (samples - 1)) * (worldMax - worldMin);
            const y = worldMin + (j / (samples - 1)) * (worldMax - worldMin);
            const l = lossFn(x, y, 0); // Use unblurred landscape for consistent coloring
            if (l < minL) minL = l;
            if (l > maxL) maxL = l;
        }
    }
    return { minLoss: minL, maxLoss: maxL };
  }, [lossFn, settings, landscapeParams]); // Re-calc if landscape changes

  // Descent state
  const posRef = useRef({ x: 3.6, y: -3.4 });
  const velRef = useRef({ vx: 0, vy: 0 }); // For SGD
  const adamMRef = useRef({ x: 0, y: 0 }); // For Adam
  const adamVRef = useRef({ x: 0, y: 0 }); // For Adam
  const trailRef = useRef<{ x: number; y: number }[]>([]);
  const stepCountRef = useRef(0);
  const startTimeRef = useRef<number | null>(null);

  // World <-> screen mapping
  const mapFns = useMemo(() => {
    function worldToScreen(
      x: number,
      y: number,
      canvas: HTMLCanvasElement,
      currentBlur: number,
    ): { sx: number; sy: number } {
      const paddings = { top: 24, right: 24, bottom: 24, left: 24 };
      const w = canvas.clientWidth - (paddings.left + paddings.right);
      const h = canvas.clientHeight - (paddings.top + paddings.bottom);
      const { worldMin, worldMax, heightScale } = settings;
      const nx = (x - worldMin) / (worldMax - worldMin);
      const ny = (y - worldMin) / (worldMax - worldMin);
      const baseY = paddings.top + ny * h;
      const sx = paddings.left + nx * w;
      const z = lossFn(x, y, currentBlur);
      const sy = baseY - z * heightScale; // lift by height
      return { sx, sy };
    }
    return { worldToScreen };
  }, [settings, lossFn]);
  
  const drawOnce = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Show un-blurred landscape if idle, otherwise apply continuation blur.
    let currentBlur = 0;
    if (isRunning || stepCountRef.current > 0) {
        const totalSteps = Math.max(600, Math.floor(60 * cycleSeconds * 5));
        const progress = Math.min(1, stepCountRef.current / totalSteps);
        currentBlur = blur * (1 - progress);
    }
    
    const { nLines, samplesPerLine, worldMin, worldMax, lineWidth } = settings;

    ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
    const grd = ctx.createLinearGradient(0, 0, 0, canvas.clientHeight);
    grd.addColorStop(0, "#06080b");
    grd.addColorStop(1, "#0a0d12");
    ctx.fillStyle = grd;
    ctx.fillRect(0, 0, canvas.clientWidth, canvas.clientHeight);

    ctx.lineWidth = lineWidth;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.globalCompositeOperation = "lighter";

    for (let i = 0; i < nLines; i++) {
        const yWorld = worldMin + ((i + 0.5) / nLines) * (worldMax - worldMin);
        let lastPt: {sx: number, sy: number, z: number} | null = null;
        for (let j = 0; j <= samplesPerLine; j++) {
            const xWorld = worldMin + (j / samplesPerLine) * (worldMax - worldMin);
            const z = lossFn(xWorld, yWorld, currentBlur);
            const { sx, sy } = mapFns.worldToScreen(xWorld, yWorld, canvas, currentBlur);

            if (lastPt) {
                const avgZ = (z + lastPt.z) / 2;
                ctx.strokeStyle = heightToColor(avgZ, minLoss, maxLoss);
                ctx.beginPath();
                ctx.moveTo(lastPt.sx, lastPt.sy);
                ctx.lineTo(sx, sy);
                ctx.stroke();
            }
            lastPt = { sx, sy, z };
        }
    }

    if (trailRef.current.length > 1) {
    ctx.globalCompositeOperation = "source-over";
    ctx.lineWidth = 2.25;
    ctx.strokeStyle = "rgba(255, 255, 255, 0.65)";
    ctx.beginPath();
    for (let i = 0; i < trailRef.current.length; i++) {
        const p = trailRef.current[i];
        const { sx, sy } = mapFns.worldToScreen(p.x, p.y, canvas, currentBlur);
        if (i === 0) ctx.moveTo(sx, sy);
        else ctx.lineTo(sx, sy);
    }
    ctx.stroke();
    }

    const { sx, sy } = mapFns.worldToScreen(posRef.current.x, posRef.current.y, canvas, currentBlur);
    const r = 8;
    const g2 = ctx.createRadialGradient(sx - 2, sy - 2, r * 0.2, sx, sy, r);
    g2.addColorStop(0, "#ffffff");
    g2.addColorStop(1, "#2dd4bf");
    ctx.fillStyle = g2;
    ctx.beginPath();
    ctx.arc(sx, sy, r, 0, Math.PI * 2);
    ctx.fill();

    const L = lossFn(posRef.current.x, posRef.current.y, currentBlur);
    ctx.fillStyle = "rgba(255,255,255,0.85)";
    ctx.font = "500 13px ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto";
    ctx.fillText(
    `x=${posRef.current.x.toFixed(2)}  y=${posRef.current.y.toFixed(2)}  loss=${L.toFixed(3)}`, 16, 24);
}, [mapFns, settings, lossFn, minLoss, maxLoss, blur, cycleSeconds, isRunning]);

  // Resize handling
  useEffect(() => {
    const resize = () => {
      const canvas = canvasRef.current;
      const parent = containerRef.current;
      if (!canvas || !parent) return;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = parent.clientWidth;
      const h = Math.max(420, Math.floor(parent.clientWidth * 0.6));
      canvas.width = Math.floor(w * dpr);
      canvas.height = Math.floor(h * dpr);
      canvas.style.width = w + "px";
      canvas.style.height = h + "px";
      const ctx = canvas.getContext("2d");
      if (ctx) ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      drawOnce();
    };
    let ro: ResizeObserver;
    if (containerRef.current) {
        ro = new ResizeObserver(resize);
        ro.observe(containerRef.current);
    }
    resize();
    return () => { if(ro && containerRef.current) ro.unobserve(containerRef.current); };
  }, [drawOnce]);
  
  const stepDescent = useCallback((lr: number) => {
    const totalSteps = Math.max(600, Math.floor(60 * cycleSeconds * 5));
    const progress = Math.min(1, stepCountRef.current / totalSteps);
    
    const currentBlur = blur * (1 - progress);
    const currentTemp = annealTemp ? temperature * (1 - progress) : temperature;
    
    let { gx, gy } = gradLoss(posRef.current.x, posRef.current.y, currentBlur);

    // Langevin Dynamics thermal noise
    if (currentTemp > 0) {
        const thermalNoise = Math.sqrt(2 * currentTemp / (lr + 1e-8));
        gx += thermalNoise * randomNormal();
        gy += thermalNoise * randomNormal();
    }
    
    if (optimizer === 'sgd') {
        const { x, y } = posRef.current;
        let lookX = x, lookY = y;
        if (nesterov) {
            lookX += momentum * velRef.current.vx;
            lookY += momentum * velRef.current.vy;
            ({ gx, gy } = gradLoss(lookX, lookY, currentBlur)); // Re-evaluate gradient at lookahead
        }
        velRef.current.vx = momentum * velRef.current.vx - lr * gx;
        velRef.current.vy = momentum * velRef.current.vy - lr * gy;
        posRef.current.x += velRef.current.vx;
        posRef.current.y += velRef.current.vy;
    } else { // Adam
        const beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8;
        const m = adamMRef.current;
        const v = adamVRef.current;
        const t = stepCountRef.current + 1;

        m.x = beta1 * m.x + (1 - beta1) * gx;
        m.y = beta1 * m.y + (1 - beta1) * gy;
        v.x = beta2 * v.x + (1 - beta2) * (gx * gx);
        v.y = beta2 * v.y + (1 - beta2) * (gy * gy);
        
        const m_hat_x = m.x / (1 - Math.pow(beta1, t));
        const m_hat_y = m.y / (1 - Math.pow(beta1, t));
        const v_hat_x = v.x / (1 - Math.pow(beta2, t));
        const v_hat_y = v.y / (1 - Math.pow(beta2, t));
        
        posRef.current.x -= lr * m_hat_x / (Math.sqrt(v_hat_x) + epsilon);
        posRef.current.y -= lr * m_hat_y / (Math.sqrt(v_hat_y) + epsilon);
    }

    const { worldMin, worldMax } = settings;
    posRef.current.x = Math.max(worldMin, Math.min(worldMax, posRef.current.x));
    posRef.current.y = Math.max(worldMin, Math.min(worldMax, posRef.current.y));

    trailRef.current.push({ x: posRef.current.x, y: posRef.current.y });
    if (trailRef.current.length > 600) trailRef.current.shift();
    }, [momentum, nesterov, settings, gradLoss, optimizer, cycleSeconds, blur, annealTemp, temperature]);

  // Core animation loop
  useEffect(() => {
    function frame(t: number) {
      if (!isRunning) return;
      if (startTimeRef.current === null) startTimeRef.current = t;
      const stepsPerCycle = Math.max(6, Math.floor(60 * cycleSeconds));

      const substeps = 2; // smooth motion
      for (let s = 0; s < substeps; s++) {
        const lr = lrSchedule(schedule, baseLR, stepCountRef.current, stepsPerCycle);
        stepDescent(lr);
        stepCountRef.current += 1;
      }

      drawOnce();
      rafRef.current = requestAnimationFrame(frame);
    }

    if (isRunning) { rafRef.current = requestAnimationFrame(frame); }
    return () => { if (rafRef.current) cancelAnimationFrame(rafRef.current); rafRef.current = null; };
  }, [isRunning, baseLR, cycleSeconds, drawOnce, schedule, stepDescent]);

  const resetRandom = useCallback(() => {
    setIsRunning(false);
    const { worldMin, worldMax } = settings;
    posRef.current = { x: rand(worldMin, worldMax), y: rand(worldMin, worldMax) };
    velRef.current = { vx: 0, vy: 0 };
    adamMRef.current = { x: 0, y: 0 };
    adamVRef.current = { x: 0, y: 0 };
    trailRef.current = [];
    stepCountRef.current = 0;
    startTimeRef.current = null;
    drawOnce();
  }, [settings, drawOnce]);


  function stepOnce() {
    if (isRunning) return;
    const stepsPerCycle = Math.max(6, Math.floor(60 * cycleSeconds));
    const lr = lrSchedule(schedule, baseLR, stepCountRef.current, stepsPerCycle);
    stepDescent(lr);
    stepCountRef.current += 1;
    drawOnce();
  }

  const handleRegenerate = useCallback(() => {
    setLandscapeParams(generateRandomParams());
    resetRandom();
  }, [resetRandom]);

  return (
    <div className="w-full max-w-5xl mx-auto flex flex-col gap-4">
      {/* Control panel */}
      <div className="w-full">
          <div className="flex flex-col gap-3 rounded-xl bg-black/50 p-3 text-[12px] text-white backdrop-blur-lg border border-white/20 shadow-lg">
              <div className="flex flex-wrap items-center justify-center gap-2">
                <button onClick={() => setIsRunning((v) => !v)} className="rounded-lg bg-emerald-500/90 px-3 py-1.5 font-semibold hover:bg-emerald-500 min-w-[70px]">{isRunning ? "Pause" : "Run"}</button>
                <button onClick={stepOnce} disabled={isRunning} className="rounded-lg bg-sky-500/90 px-3 py-1.5 font-semibold hover:bg-sky-500 disabled:bg-gray-500 disabled:cursor-not-allowed">Step</button>
                <button onClick={resetRandom} className="rounded-lg bg-rose-500/90 px-3 py-1.5 font-semibold hover:bg-rose-500">Reset Ball</button>
                <button onClick={handleRegenerate} className="rounded-lg bg-violet-500/90 px-3 py-1.5 font-semibold hover:bg-violet-500">New Landscape</button>
                 <div className="mx-2 h-6 w-px bg-white/30 hidden sm:block" />
                 <label className="flex items-center gap-2">
                    <span className="opacity-80">Optimizer</span>
                    <select value={optimizer} onChange={(e) => setOptimizer(e.target.value as 'sgd' | 'adam')} className="rounded-md bg-white/10 px-2 py-1">
                        <option value="adam" className="bg-gray-800">Adam</option>
                        <option value="sgd" className="bg-gray-800">SGD+Momentum</option>
                    </select>
                </label>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-x-4 gap-y-2 border-t border-white/20 pt-3">
                <label className="flex items-center gap-2" title="The base learning rate for the optimizer.">
                    <span className="opacity-80 w-12">LR</span>
                    <input type="range" min={0.001} max={optimizer === 'adam' ? 0.1 : 0.35} step={0.001} value={baseLR} onChange={(e) => setBaseLR(parseFloat(e.target.value))} className="w-24" />
                    <span className="tabular-nums w-12 text-left">{baseLR.toFixed(3)}</span>
                </label>
                 <label className={`flex items-center gap-2 ${optimizer !== 'sgd' && 'opacity-50 cursor-not-allowed'}`} title="Momentum factor (only for SGD).">
                    <span className="opacity-80 w-12">Momentum</span>
                    <input type="range" min={0} max={0.99} step={0.01} value={momentum} onChange={(e) => setMomentum(parseFloat(e.target.value))} className="w-24" disabled={optimizer !== 'sgd'} />
                    <span className="tabular-nums w-12 text-left">{momentum.toFixed(2)}</span>
                </label>
                <label className="flex items-center gap-2" title="Adds thermal noise to escape local minima (Langevin Dynamics).">
                    <span className="opacity-80 w-12">Temp.</span>
                    <input type="range" min={0} max={1.0} step={0.01} value={temperature} onChange={(e) => setTemperature(parseFloat(e.target.value))} className="w-24" />
                    <span className="tabular-nums w-12 text-left">{temperature.toFixed(2)}</span>
                </label>
                 <label className="flex items-center gap-2" title="Blurs the landscape to find better regions, then sharpens it (Continuation).">
                    <span className="opacity-80 w-12">Blur</span>
                    <input type="range" min={0} max={1.0} step={0.01} value={blur} onChange={(e) => setBlur(parseFloat(e.target.value))} className="w-24" />
                    <span className="tabular-nums w-12 text-left">{blur.toFixed(2)}</span>
                </label>
                 <label className="flex items-center gap-2">
                    <span className="opacity-80 w-12">Schedule</span>
                     <select value={schedule} onChange={(e) => setSchedule(e.target.value as ScheduleKind)} className="rounded-md bg-white/10 px-2 py-1 flex-1">
                        {SCHEDULES.map((k) => (<option key={k} value={k} className="bg-gray-800">{k}</option>))}
                    </select>
                </label>
                 <label className="flex items-center gap-2" title="The length of a learning rate schedule cycle in seconds.">
                    <span className="opacity-80 w-12">Cycle (s)</span>
                    <input type="range" min={2} max={14} step={1} value={cycleSeconds} onChange={(e) => setCycleSeconds(parseFloat(e.target.value))} className="w-24" />
                    <span className="tabular-nums w-12 text-left">{cycleSeconds}</span>
                </label>
                <div className="flex items-center gap-4">
                     <label className="flex items-center gap-2" title="Gradually decrease temperature to zero over the run.">
                        <input type="checkbox" checked={annealTemp} onChange={(e) => setAnnealTemp(e.target.checked)} />
                        <span className="opacity-80">Anneal</span>
                    </label>
                    <label className={`flex items-center gap-2 ${optimizer !== 'sgd' && 'opacity-50 cursor-not-allowed'}`} title="Use Nesterov accelerated gradient (only for SGD).">
                        <input type="checkbox" checked={nesterov} onChange={(e) => setNesterov(e.target.checked)} disabled={optimizer !== 'sgd'}/>
                        <span className="opacity-80">Nesterov</span>
                    </label>
                </div>
              </div>
            </div>
      </div>
       <div ref={containerRef} className="w-full select-none">
            <canvas ref={canvasRef} className="block w-full rounded-2xl shadow-lg bg-black/50 border border-white/20" />
        </div>
    </div>
  );
}