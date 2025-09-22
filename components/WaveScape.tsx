// FIX: Add `useCallback` to the import from 'react' to resolve 'Cannot find name' errors.
import React, { useEffect, useMemo, useRef, useState, useCallback } from "react";

// WaveScape Gradient Descent — single-file React component
// Drop this file into your Next.js app (e.g., app/components/WaveScape.tsx)
// and render <WaveScape/> anywhere. No external deps. Tailwind optional.

// ------------------------------
// Math — loss landscape
// ------------------------------
// A handcrafted non-convex loss with saddles + local minima.
// Coordinates live in world space W = [-5, 5] x [-5, 5].
function lossFn(x: number, y: number): number {
  // Radial ripples
  const r2 = x * x + y * y;
  const term1 = 0.45 * Math.cos(0.8 * r2);
  // Anisotropic bowl so gradients don’t explode at the edges
  const term2 = 0.06 * (0.8 * x * x + 0.35 * y * y);
  // Sinusoidal ridges and saddles
  const term3 = 0.9 * Math.sin(1.1 * x) * Math.sin(0.7 * y);
  const term4 = 0.5 * Math.cos(1.7 * x + 0.6 * y);
  // Two uneven bumps to seed local structure
  const term5 = 1.2 * Math.exp(-((x - 1.8) ** 2 + (y + 1.3) ** 2) / 0.9);
  const term6 = 0.9 * Math.exp(-((x + 1.4) ** 2 + (y - 2.3) ** 2) / 0.7);
  return term1 + term2 + term3 + term4 + term5 + term6;
}

function gradLoss(x: number, y: number): { gx: number; gy: number } {
  // Finite differences (central). h tuned to world extents.
  const h = 1e-3 * 10; // domain width ~10
  const gx = (lossFn(x + h, y) - lossFn(x - h, y)) / (2 * h);
  const gy = (lossFn(x, y + h) - lossFn(x, y - h)) / (2 * h);
  return { gx, gy };
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
// Palette helper — HSL across lines
// ------------------------------
function lineColour(i: number, n: number, alpha = 0.85): string {
  const hue = (320 * (i / n) + 20) % 360; // rich multi-colour spread
  const sat = 80;
  const light = 58;
  return `hsla(${hue}, ${sat}%, ${light}%, ${alpha})`;
}

// ------------------------------
// Component
// ------------------------------
export const WaveScape: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const rafRef = useRef<number | null>(null);
  const [isRunning, setIsRunning] = useState(true);

  // Simulation parameters
  const [baseLR, setBaseLR] = useState(0.08);
  const [momentum, setMomentum] = useState(0.85);
  const [nesterov, setNesterov] = useState(true);
  const [schedule, setSchedule] = useState<ScheduleKind>("cosine-restarts");
  const [cycleSeconds, setCycleSeconds] = useState(6);

  // Landscape drawing details
  const settings = useMemo(
    () => ({
      worldMin: -5,
      worldMax: 5,
      nLines: 46, // number of ridge-lines
      samplesPerLine: 320,
      heightScale: 36, // vertical exaggeration in px per world unit z
      lineWidth: 1.1,
    }),
    []
  );

  // Descent state
  const posRef = useRef({ x: 3.6, y: -3.4 });
  const velRef = useRef({ vx: 0, vy: 0 });
  const trailRef = useRef<{ x: number; y: number }[]>([]);
  const stepCountRef = useRef(0);
  const startTimeRef = useRef<number | null>(null);

  // Resize handling for crisp canvas
  useEffect(() => {
    const resize = () => {
      const canvas = canvasRef.current;
      const parent = containerRef.current;
      if (!canvas || !parent) return;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = parent.clientWidth;
      const h = Math.max(420, Math.floor(parent.clientWidth * 0.45));
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
    
    return () => {
        if(ro && containerRef.current) {
            ro.unobserve(containerRef.current);
        }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // World <-> screen mapping
  const mapFns = useMemo(() => {
    function worldToScreen(
      x: number,
      y: number,
      canvas: HTMLCanvasElement
    ): { sx: number; sy: number } {
      const padding = 24;
      const w = canvas.clientWidth - padding * 2;
      const h = canvas.clientHeight - padding * 2;
      const { worldMin, worldMax, heightScale } = settings;
      const nx = (x - worldMin) / (worldMax - worldMin);
      const ny = (y - worldMin) / (worldMax - worldMin);
      const baseY = padding + ny * h;
      const sx = padding + nx * w;
      const z = lossFn(x, y);
      const sy = baseY - z * heightScale; // lift by height
      return { sx, sy };
    }

    function screenXfromWorld(x: number, canvas: HTMLCanvasElement): number {
      const padding = 24;
      const w = canvas.clientWidth - padding * 2;
      const nx = (x - settings.worldMin) / (settings.worldMax - settings.worldMin);
      return padding + nx * w;
    }

    function baseYfromWorld(y: number, canvas: HTMLCanvasElement): number {
      const padding = 24;
      const h = canvas.clientHeight - padding * 2;
      const ny = (y - settings.worldMin) / (settings.worldMax - settings.worldMin);
      return padding + ny * h;
    }

    return { worldToScreen, screenXfromWorld, baseYfromWorld };
  }, [settings]);

    const stepDescent = useCallback((lr: number) => {
        const { x, y } = posRef.current;
        let gx: number, gy: number;

        if (nesterov) {
        // look-ahead gradient
        const lookX = x + momentum * velRef.current.vx;
        const lookY = y + momentum * velRef.current.vy;
        ({ gx, gy } = gradLoss(lookX, lookY));
        } else {
        ({ gx, gy } = gradLoss(x, y));
        }

        // Momentum update
        velRef.current.vx = momentum * velRef.current.vx - lr * gx;
        velRef.current.vy = momentum * velRef.current.vy - lr * gy;

        // Position update
        posRef.current.x += velRef.current.vx;
        posRef.current.y += velRef.current.vy;

        // Constrain to world bounds so the ball doesn’t run away
        const { worldMin, worldMax } = settings;
        posRef.current.x = Math.max(worldMin, Math.min(worldMax, posRef.current.x));
        posRef.current.y = Math.max(worldMin, Math.min(worldMax, posRef.current.y));

        // Update trail
        const pt = { x: posRef.current.x, y: posRef.current.y };
        trailRef.current.push(pt);
        if (trailRef.current.length > 600) trailRef.current.shift();
    }, [momentum, nesterov, settings]);
    
    const drawOnce = useCallback(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;

        const { nLines, samplesPerLine, worldMin, worldMax, heightScale, lineWidth } =
        settings;

        // Clear
        ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
        // Subtle background
        const grd = ctx.createLinearGradient(0, 0, 0, canvas.clientHeight);
        grd.addColorStop(0, "#06080b");
        grd.addColorStop(1, "#0a0d12");
        ctx.fillStyle = grd;
        ctx.fillRect(0, 0, canvas.clientWidth, canvas.clientHeight);

        ctx.lineWidth = lineWidth;
        ctx.lineJoin = "round";
        ctx.lineCap = "round";
        ctx.globalCompositeOperation = "lighter";

        // Draw multi-colour wave lines (like stacked mountain ridges)
        for (let i = 0; i < nLines; i++) {
        const yWorld = worldMin + ((i + 0.5) / nLines) * (worldMax - worldMin);
        const baseY = mapFns.baseYfromWorld(yWorld, canvas);
        ctx.strokeStyle = lineColour(i, nLines, 0.8);
        ctx.beginPath();

        for (let j = 0; j <= samplesPerLine; j++) {
            const xWorld = worldMin + (j / samplesPerLine) * (worldMax - worldMin);
            const z = lossFn(xWorld, yWorld);
            const sx = mapFns.screenXfromWorld(xWorld, canvas);
            const sy = baseY - z * heightScale;
            if (j === 0) ctx.moveTo(sx, sy);
            else ctx.lineTo(sx, sy);
        }
        ctx.stroke();
        }

        // Path trail
        if (trailRef.current.length > 1) {
        ctx.globalCompositeOperation = "source-over";
        ctx.lineWidth = 2.25;
        ctx.strokeStyle = "rgba(255, 255, 255, 0.65)";
        ctx.beginPath();
        for (let i = 0; i < trailRef.current.length; i++) {
            const p = trailRef.current[i];
            const { sx, sy } = mapFns.worldToScreen(p.x, p.y, canvas);
            if (i === 0) ctx.moveTo(sx, sy);
            else ctx.lineTo(sx, sy);
        }
        ctx.stroke();
        }

        // The "ball" (optimizer point)
        const { sx, sy } = mapFns.worldToScreen(posRef.current.x, posRef.current.y, canvas);
        const r = 8;
        const g2 = ctx.createRadialGradient(sx - 2, sy - 2, r * 0.2, sx, sy, r);
        g2.addColorStop(0, "#ffffff");
        g2.addColorStop(1, "#2dd4bf"); // teal-ish
        ctx.fillStyle = g2;
        ctx.beginPath();
        ctx.arc(sx, sy, r, 0, Math.PI * 2);
        ctx.fill();

        // HUD: loss value
        const L = lossFn(posRef.current.x, posRef.current.y);
        ctx.fillStyle = "rgba(255,255,255,0.85)";
        ctx.font = "500 13px ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto";
        ctx.fillText(
        `x=${posRef.current.x.toFixed(2)}  y=${posRef.current.y.toFixed(2)}  loss=${L.toFixed(
            3
        )}`,
        16,
        24
        );
    }, [mapFns, settings]);

  // Core animation loop
  useEffect(() => {
    function frame(t: number) {
      if (!isRunning) return;
      if (startTimeRef.current === null) startTimeRef.current = t;
      const stepsPerCycle = Math.max(6, Math.floor(60 * cycleSeconds));

      // physics step (fixed dt for stability)
      const substeps = 2; // smooth motion
      for (let s = 0; s < substeps; s++) {
        const lr = lrSchedule(
          schedule,
          baseLR,
          stepCountRef.current,
          stepsPerCycle
        );
        stepDescent(lr);
        stepCountRef.current += 1;
      }

      drawOnce();
      rafRef.current = requestAnimationFrame(frame);
    }

    if (isRunning) {
        rafRef.current = requestAnimationFrame(frame);
    }
    
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    };
  }, [isRunning, baseLR, cycleSeconds, drawOnce, schedule, stepDescent]);

  function resetRandom() {
    const { worldMin, worldMax } = settings;
    const rand = (a: number, b: number) => a + Math.random() * (b - a);
    posRef.current = { x: rand(worldMin, worldMax), y: rand(worldMin, worldMax) };
    velRef.current = { vx: 0, vy: 0 };
    trailRef.current = [];
    stepCountRef.current = 0;
    startTimeRef.current = null;
    drawOnce();
  }

  function stepOnce() {
    const stepsPerCycle = Math.max(6, Math.floor(60 * cycleSeconds));
    const lr = lrSchedule(schedule, baseLR, stepCountRef.current, stepsPerCycle);
    stepDescent(lr);
    stepCountRef.current += 1;
    drawOnce();
  }

  return (
    <div ref={containerRef} className="relative w-full select-none">
      <canvas ref={canvasRef} className="block w-full rounded-2xl shadow-lg bg-black/50" />

      {/* Control panel */}
      <div className="flex flex-col sm:flex-row absolute left-1/2 -translate-x-1/2 top-4 w-[calc(100%-2rem)]">
          <div className="pointer-events-auto flex max-w-full flex-wrap items-center justify-center gap-x-2 gap-y-2 rounded-xl bg-black/50 p-2 text-[12px] text-white backdrop-blur-lg mx-auto">
                <button
                onClick={() => setIsRunning((v) => !v)}
                className="rounded-lg bg-emerald-500/90 px-3 py-1.5 font-semibold hover:bg-emerald-500"
                >
                {isRunning ? "Pause" : "Run"}
                </button>
                <button
                onClick={stepOnce}
                disabled={isRunning}
                className="rounded-lg bg-sky-500/90 px-3 py-1.5 font-semibold hover:bg-sky-500 disabled:bg-gray-500 disabled:cursor-not-allowed"
                >
                Step
                </button>
                <button
                onClick={resetRandom}
                className="rounded-lg bg-rose-500/90 px-3 py-1.5 font-semibold hover:bg-rose-500"
                >
                Reset
                </button>

                <div className="mx-2 h-6 w-px bg-white/30 hidden md:block" />

                <label className="flex items-center gap-2">
                <span className="opacity-80">LR</span>
                <input
                    type="range"
                    min={0.005}
                    max={0.35}
                    step={0.005}
                    value={baseLR}
                    onChange={(e) => setBaseLR(parseFloat(e.target.value))}
                    className="w-20"
                />
                <span className="tabular-nums w-12 text-left">{baseLR.toFixed(3)}</span>
                </label>

                <label className="flex items-center gap-2">
                <span className="opacity-80">Momentum</span>
                <input
                    type="range"
                    min={0}
                    max={0.99}
                    step={0.01}
                    value={momentum}
                    onChange={(e) => setMomentum(parseFloat(e.target.value))}
                    className="w-20"
                />
                <span className="tabular-nums w-10 text-left">{momentum.toFixed(2)}</span>
                </label>

                <label className="flex items-center gap-2">
                <input
                    type="checkbox"
                    checked={nesterov}
                    onChange={(e) => setNesterov(e.target.checked)}
                />
                <span className="opacity-80">Nesterov</span>
                </label>

                <div className="mx-2 h-6 w-px bg-white/30 hidden lg:block" />

                <label className="flex items-center gap-2">
                <span className="opacity-80">Schedule</span>
                <select
                    className="rounded-md bg-white/10 px-2 py-1"
                    value={schedule}
                    onChange={(e) => setSchedule(e.target.value as ScheduleKind)}
                >
                    {SCHEDULES.map((k) => (
                    <option key={k} value={k} className="bg-gray-800">
                        {k}
                    </option>
                    ))}
                </select>
                </label>

                <label className="flex items-center gap-2">
                <span className="opacity-80">Cycle (s)</span>
                <input
                    type="range"
                    min={2}
                    max={14}
                    step={1}
                    value={cycleSeconds}
                    onChange={(e) => setCycleSeconds(parseFloat(e.target.value))}
                    className="w-20"
                />
                <span className="tabular-nums w-8 text-left">{cycleSeconds}</span>
                </label>
            </div>
      </div>
    </div>
  );
}