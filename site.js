
(function(){
"use strict";
const FX = {
  eq:["EQ","Equalizer","Raises or lowers chosen frequency ranges, for example less boom or more clarity."],
  compressor:["Comp","Compressor","Turns down the loudest moments so the level is steadier and fuller."],
  noisegate:["Gate","Noise gate","Mutes the track when it is quiet, removing bleed and hum between notes."],
  gain_panning:["Gain/Pan","Gain and panning","Sets how loud a part is and where it sits between left and right."],
  stereo_imager:["Width","Stereo imager","Makes a sound wider or narrower in the stereo field."],
  distortion:["Drive","Distortion","Adds saturation and grit by overdriving the signal."],
  delay:["Delay","Delay","Adds echoes that repeat after a short time."],
  reverb:["Reverb","Reverb","Adds the sense of a room or hall around the sound."],
  limiter:["Limiter","Limiter","Stops the final mix from going over a peak ceiling."]
};
const ICON = {
  eq:'<path d="M2 20 C10 20 12 6 20 6 S30 22 42 14" fill="none" stroke="currentColor" stroke-width="2.4"/>',
  compressor:'<path d="M2 15 L8 4 L12 26 L18 9 L22 21 L26 11 L30 19 L34 12 L38 18 L42 14" fill="none" stroke="currentColor" stroke-width="2"/><path d="M2 8 H42 M2 22 H42" stroke="currentColor" stroke-width="1" stroke-dasharray="2 2"/>',
  noisegate:'<path d="M2 15 H10 L13 5 L16 25 L19 8 L22 15 H42" fill="none" stroke="currentColor" stroke-width="2"/><rect x="27" y="10" width="13" height="10" fill="none" stroke="currentColor" stroke-width="1.6"/>',
  gain_panning:'<path d="M8 4 V26 M30 4 V26" stroke="currentColor" stroke-width="2"/><rect x="4" y="14" width="8" height="5" rx="1" fill="currentColor"/><rect x="26" y="8" width="8" height="5" rx="1" fill="currentColor"/>',
  stereo_imager:'<path d="M22 15 L4 6 M22 15 L4 24 M22 15 L40 6 M22 15 L40 24" stroke="currentColor" stroke-width="2"/><circle cx="22" cy="15" r="3" fill="currentColor"/>',
  distortion:'<path d="M2 15 L8 4 H14 L18 26 H24 L28 4 H34 L38 26 H42" fill="none" stroke="currentColor" stroke-width="2"/>',
  delay:'<rect x="3" y="8" width="6" height="14" rx="1" fill="currentColor"/><rect x="16" y="10" width="5" height="10" rx="1" fill="currentColor" opacity=".7"/><rect x="27" y="12" width="4" height="6" rx="1" fill="currentColor" opacity=".45"/><rect x="36" y="13" width="3" height="4" rx="1" fill="currentColor" opacity=".25"/>',
  reverb:'<path d="M6 15 a4 4 0 0 1 0 0" stroke="currentColor"/><circle cx="8" cy="15" r="3" fill="currentColor"/><path d="M16 7 a10 10 0 0 1 0 16 M24 3 a16 16 0 0 1 0 24 M32 1 a20 20 0 0 1 0 28" fill="none" stroke="currentColor" stroke-width="1.8"/>',
  limiter:'<path d="M2 22 L8 12 L12 18 L17 6 L22 6 L26 6 L30 16 L36 6 L42 14" fill="none" stroke="currentColor" stroke-width="2"/><path d="M2 6 H42" stroke="currentColor" stroke-width="1.4"/>'
};
const COND = {
  reference:["Style reference","Another part of the same song, the sound to aim for."],
  pro:["Professional mix","The engineer's mix of these 15 seconds (hidden from all systems)."],
  mixagent:["MixAgent","Hierarchical plan with learned detector and critic."],
  gemini:["Gemini 3.8 Flash","Single pass, same inputs and effects."],
  gpt4o:["GPT-4o-audio","Single pass, same inputs and effects."],
  qwen:["Qwen3-Omni","Single pass, same inputs and effects."],
  drysum:["Dry sum","Tracks added together, no effects."],
  mst:["MST","Mixing style transfer, single pass."],
  dmst:["Diff-MST","Differentiable mixing style transfer, single pass."],
  stemfx:["StemFX","Effect-chain prediction on separated stems, single pass."]
};
const COND_REAL = Object.assign({}, COND, {
  reference:["Style reference","A different song of the same genre, the sound to aim for."],
  pro:["Song's own mix","The engineer's mix of this song. Shown as an anchor, not the target."]
});
const STEM4 = new Set(["mst","stemfx"]);
const ROW_PRIOR = ["mst","dmst","stemfx"];
const DSNAME = {medleydb:"MedleyDB", albumdb:"AlbumDB", real:"Listening test"};
const ROW_TARGETS = ["reference","pro"];
const ROW_SYSTEMS = ["mixagent","gemini","gpt4o","qwen"];
const BANDS = ["Low","Lo-mid","Hi-mid","High"];

const NOTES = {
  medleydb:{
    sample_181:{tags:"Vocal tracks ×4 · Piano ×2",
      note:"The piano's low notes and the space around the voices are the main differences to compare. Gemini and GPT-4o-audio have less bass overall, while MixAgent gives the recording a fuller sound. The music also extends farther to the left and right in MixAgent, closer to the professional mix."},
    sample_130:{tags:"Trumpet · Trombone · Tenor sax · Bass clarinet · Double bass ×2 · Drums ×5 · Piano ×4",
      note:"The double bass plays low notes beneath the horns and cymbals. MixAgent stays close to the professional mix's bass level, while Gemini, GPT-4o-audio and Qwen3-Omni shift the balance toward the higher sounds. There is still a clear gap in width. The professional mix spreads the band much farther to the left and right."}
  },
  albumdb:{
    sample_013:{tags:"Cello · Viola · Violins ×2 · Acoustic guitars L/R · Bass · Lead vocal",
      note:"The violins' sharp edge matters here, especially alongside the singer and acoustic guitars. Gemini, GPT-4o-audio and Qwen3-Omni strongly emphasize the frequency range that can make strings sound scratchy. MixAgent stays closer to the professional mix in this range and gives the group more space from left to right."},
    sample_003:{tags:"Drums · Bass · Acoustic and electric guitars ×5 · Shaker · Backing vocals",
      note:"Cymbal sizzle and the shaker's rattle are the details to compare in this clip. Gemini and Qwen3-Omni strongly emphasize the high frequencies, with GPT-4o-audio doing so to a lesser extent. MixAgent is much closer to the professional mix in this range, with less emphasis on those bright sounds."},
    sample_009:{tags:"Drums · Bass · Electric guitars ×3 · Organ · Lead vocal",
      note:"With electric guitars, organ, drums and a lead singer sharing the chorus, the width of the whole band is a useful comparison. MixAgent nearly matches the professional mix's spread from left to right. GPT-4o-audio and Qwen3-Omni bring the sound closer to the center, while Gemini puts more emphasis on the high frequencies, where cymbal sizzle is heard."},
  },
  real:{
    pair_007:{tags:"12 tracks: drums ×6 · guitars ×2 · bass ×2 · vocals ×2",
      note:"Bass and drums provide the low notes beneath the guitars and singing. GPT-4o-audio has less bass overall, while Gemini pushes the balance toward the high sounds. MixAgent's balance is closer to the reference song, although the band occupies a much narrower space from left to right. Average listener ratings out of 100 were 74.9 for MixAgent, 31.8 for Gemini, 42.1 for GPT-4o-audio and 37.0 for Qwen3-Omni."},
    pair_029:{tags:"25 tracks including lap steel ×8 and melodica ×2",
      note:"The voices sit alongside lap steel guitar and melodica, a small keyboard instrument played by blowing air into it. In this clip, MixAgent, Gemini, GPT-4o-audio and Qwen3-Omni all add much more high-frequency sound than the reference song. MixAgent makes the smallest change in that range and keeps the bass level close to the reference. Qwen3-Omni has both the strongest high frequencies and less bass. Average listener ratings out of 100 were 51.5 for MixAgent, 27.6 for Gemini, 26.7 for GPT-4o-audio and 21.0 for Qwen3-Omni."},
    pair_004:{tags:"12 tracks: drums ×6 · guitars ×2 · vocals ×2 · bass ×2",
      note:"The repeating guitar part and drum beat are the main points to compare across versions. In MixAgent, the music extends farther to the left and right. Gemini, GPT-4o-audio and Qwen3-Omni gather more of the sound in the center. Although MixAgent adds less high-frequency sound than Gemini, GPT-4o-audio and Qwen3-Omni, it is still much brighter than the reference song. Average listener ratings out of 100 were 55.6 for MixAgent, 34.2 for Gemini, 30.7 for GPT-4o-audio and 38.4 for Qwen3-Omni."}
  }
};
function chip(t){
  const f = FX[t]; const name = f ? f[0] : t;
  return `<span class="chip" style="color:var(--fx-${t},var(--ink-2));border-color:color-mix(in srgb,var(--fx-${t},var(--ink-3)) 45%,transparent)"><i style="background:var(--fx-${t},var(--ink-3))"></i>${name}</span>`;
}
function chips(arr){ return arr.length ? arr.map(chip).join('<span class="arrow">→</span>') : '<span class="muted small">no effects</span>'; }
function esc(s){ return String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
// small chart: four band offsets (dB vs target) + a width meter
function meter(feat, isTarget, tgtName){
  const W=214,H=86, x0=38, bw=36, gap=8, mid=30, sc=2.0; // 2 px per dB, clipped at ±12 dB
  let g = `<line class="grid" x1="${x0-4}" x2="${W-6}" y1="${mid-24}" y2="${mid-24}"/><line class="grid" x1="${x0-4}" x2="${W-6}" y1="${mid+24}" y2="${mid+24}"/>`;
  g += `<text x="${x0-6}" y="${mid-21}" text-anchor="end">+12</text><text x="${x0-6}" y="${mid+3}" text-anchor="end">0</text><text x="${x0-6}" y="${mid+27}" text-anchor="end">−12</text>`;
  feat.bands.forEach((v,i)=>{
    const x = x0 + i*(bw+gap); const c = Math.max(-12, Math.min(12, v)); const h = Math.abs(c)*sc;
    const col = isTarget ? 'var(--bar-ok)' : (Math.abs(v) < 3 ? 'var(--bar-ok)' : (v > 0 ? 'var(--bar-pos)' : 'var(--bar-neg)'));
    const y = c >= 0 ? mid - h : mid;
    g += `<rect x="${x}" y="${y.toFixed(1)}" width="${bw}" height="${Math.max(h,1).toFixed(1)}" rx="1.5" fill="${col}"><title>${BANDS[i]}: ${v>0?'+':''}${v} dB vs ${tgtName}</title></rect>`;
    g += `<text x="${x+bw/2}" y="${mid+38}" text-anchor="middle">${BANDS[i]}</text>`;
  });
  g += `<line class="zero" x1="${x0-4}" x2="${W-6}" y1="${mid}" y2="${mid}"/>`;
  // width meter, 0 .. 0.6
  const wy = 76, wx0 = x0, wx1 = W-6, ws = v => wx0 + Math.min(v,0.6)/0.6*(wx1-wx0);
  g += `<text x="${x0-6}" y="${wy+3}" text-anchor="end">Width</text>`;
  g += `<rect x="${wx0}" y="${wy-3}" width="${wx1-wx0}" height="6" rx="3" fill="var(--wash)" stroke="var(--line)" stroke-width=".6"/>`;
  g += `<rect x="${wx0}" y="${wy-3}" width="${(ws(feat.width)-wx0).toFixed(1)}" height="6" rx="3" fill="var(--ink-3)"/>`;
  g += `<line x1="${ws(feat.width_t).toFixed(1)}" x2="${ws(feat.width_t).toFixed(1)}" y1="${wy-6}" y2="${wy+6}" stroke="var(--accent)" stroke-width="2"><title>${tgtName} width</title></line>`;
  return `<svg class="meter" viewBox="0 0 ${W} ${H}" role="img" aria-label="Band levels ${feat.bands.map((v,i)=>BANDS[i]+' '+v+' dB').join(', ')} relative to the ${tgtName}; stereo width ${feat.width} (target ${feat.width_t})">${g}</svg>`;
}

// audio player: play button + waveform (peaks precomputed in peaks.js), native <audio> underneath
const fmt = t => `${Math.floor(t/60)}:${String(Math.floor(t%60)).padStart(2,'0')}`;
function waveSVG(code){
  const n = code.length, h = i => 1 + (code.charCodeAt(i)-48)/74*18.5;
  let top = '', bot = '';
  for (let i=0;i<n;i++){ top += `L${(i+.5).toFixed(1)} ${(20-h(i)).toFixed(2)}`; bot = `L${(i+.5).toFixed(1)} ${(20+h(i)).toFixed(2)}` + bot; }
  return `<svg viewBox="0 0 ${n} 40" preserveAspectRatio="none" aria-hidden="true"><path d="M0 20${top}L${n} 20${bot}Z"/></svg>`;
}
function player(src){
  const pk = (typeof PEAKS !== 'undefined' && PEAKS[src]) || [0, '0'.repeat(160)];
  return `<div class="wp"><button type="button" class="wp-b" aria-label="Play"></button><div class="wp-w" role="slider" tabindex="0" aria-label="Play or pause (arrow keys skip)" aria-valuemin="0" aria-valuemax="${Math.round(pk[0])}" aria-valuenow="0">${waveSVG(pk[1])}<div class="wp-p">${waveSVG(pk[1])}</div></div><span class="wp-t">${fmt(pk[0])}</span><audio preload="none" src="${src}"></audio></div>`;
}
(function(){
  const audioOf = el => el.closest('.wp').querySelector('audio');
  document.addEventListener('click', e => {
    const b = e.target.closest('.wp-b');
    if (b) { const a = audioOf(b); a.paused ? a.play() : a.pause(); return; }
    const w = e.target.closest('.wp-w');
    if (w) { const a = audioOf(w); a.paused ? a.play() : a.pause(); }
  });
  document.addEventListener('keydown', e => {
    const w = e.target.closest && e.target.closest('.wp-w');
    if (!w || !['ArrowLeft','ArrowRight',' ','Enter'].includes(e.key)) return;
    e.preventDefault(); const a = audioOf(w);
    if (e.key === ' ' || e.key === 'Enter') { a.paused ? a.play() : a.pause(); return; }
    if (a.readyState >= 1) a.currentTime = Math.max(0, Math.min(a.duration, a.currentTime + (e.key === 'ArrowRight' ? 2 : -2)));
  });
  const sync = e => {
    const a = e.target; if (!(a instanceof HTMLAudioElement)) return;
    const wp = a.closest('.wp'); if (!wp) return;
    const d = isFinite(a.duration) ? a.duration : 0, p = d ? a.currentTime / d : 0;
    wp.classList.toggle('playing', !a.paused);
    wp.querySelector('.wp-b').setAttribute('aria-label', a.paused ? 'Play' : 'Pause');
    wp.querySelector('.wp-p').style.clipPath = `inset(0 ${(100 - p*100).toFixed(2)}% 0 0)`;
    wp.querySelector('.wp-t').textContent = (a.paused && !a.currentTime) ? fmt(d || 0) : fmt(a.currentTime);
    wp.querySelector('.wp-w').setAttribute('aria-valuenow', Math.round(a.currentTime));
  };
  ['play','pause','ended','timeupdate','seeked','loadedmetadata'].forEach(t => document.addEventListener(t, sync, true));
})();

function stemsPanel(ex){
  const byLabel = {}; ex.stems.forEach(s => byLabel[s.label] = s);
  const used = new Set();
  let html = '';
  ex.plan.groups.forEach(g => {
    const rows = g.stems.map(s => { used.add(s.label); const st = byLabel[s.label]; return st ? `<div class="stemrow"><span>${esc(s.label)}</span>${player(st.mp3)}</div>` : ''; }).join('');
    html += `<div class="stembus"><div class="busname">${esc(g.name.replace(/_/g,' '))} subgroup</div><div class="stemlist">${rows}</div></div>`;
  });
  const rest = ex.stems.filter(s => !used.has(s.label));
  if (rest.length) html += `<div class="stembus"><div class="busname">direct to full mix</div><div class="stemlist">${rest.map(s=>`<div class="stemrow"><span>${esc(s.label)}</span>${player(s.mp3)}</div>`).join('')}</div></div>`;
  return `<section class="input-tracks"><h2>Input tracks <span class="count">${ex.stems.length}</span></h2>${html}</section>`;
}

function consoleSVG(plan, stages, log){
  const RH=30, BW=66, BH=20, ST=72, LW=150, TOP=stages?40:24;
  // flagged effects per element, from the correction log (the critic corrects exactly what the detector flagged)
  const norm = n => String(n).toLowerCase().replace(/_/g,' ').replace(/ bus$/,'').trim();
  const flags = {stem:{}, bus:{}, master:new Set()};
  (log||[]).forEach(r => { if (r.level==='master') r.tools.forEach(t => flags.master.add(t)); else (flags[r.level]||{})[r.level==='bus'?norm(r.element):r.element] = new Set(r.tools); });
  const rows=[], groups=[];
  plan.groups.forEach(g => { const idx=[]; g.stems.forEach(s => { idx.push(rows.length); rows.push(s); }); groups.push({g, idx}); });
  const direct=[]; plan.direct.forEach(s => { direct.push(rows.length); rows.push(s); });
  const y = r => TOP + r*RH + RH/2;
  const maxS = Math.max(1, ...rows.map(s => s.chain.length));
  const maxB = Math.max(0, ...plan.groups.map(g => g.chain.length));
  const sEnd = LW + maxS*ST + 6, busX = sEnd + 34, bStart = busX + 16, bEnd = bStart + maxB*ST + 6;
  const mX = bEnd + 40, mStart = mX + 16, mEnd = mStart + plan.master.length*ST + 6, outX = mEnd + 18;
  const W = outX + 30, H = TOP + rows.length*RH + 8;
  const busYs = groups.map(o => o.idx.reduce((a,r)=>a+y(r),0)/o.idx.length);
  const allY = busYs.concat(direct.map(y)); const mY = allY.reduce((a,v)=>a+v,0)/allY.length;
  const curve = (x1,y1,x2,y2) => { const dx=(x2-x1)*0.55; return `<path class="wire" d="M${x1} ${y1} C${x1+dx} ${y1}, ${x2-dx} ${y2}, ${x2} ${y2}"/>`; };
  const box = (t,x,yc,hot) => { const f = FX[t] ? FX[t][1] : t;
    return `<g${hot?' class="hot"':''}><title>${esc(f)}${hot?', flagged by the detector':''}</title>${hot?`<rect class="halo" x="${x-3}" y="${yc-BH/2-3}" width="${BW+6}" height="${BH+6}" rx="6"/>`:''}<rect x="${x}" y="${yc-BH/2}" width="${BW}" height="${BH}" rx="4" style="fill:color-mix(in srgb,var(--fx-${t},var(--ink-3)) 14%,var(--panel));stroke:var(--fx-${t},var(--ink-3))" stroke-width="1.3"/>`+
      `<svg x="${x+3}" y="${yc-5}" width="15" height="10" viewBox="0 0 44 30" style="color:var(--fx-${t},var(--ink-3))" overflow="visible">${ICON[t]||''}</svg>`+
      `<text x="${x+20}" y="${yc+3.5}" font-size="9.5" font-weight="500">${esc(FX[t]?FX[t][0]:t)}</text></g>`; };
  const sum = (x,yc) => `<circle cx="${x}" cy="${yc}" r="9" style="fill:var(--panel);stroke:var(--ink-2)" stroke-width="1.4"/><text x="${x}" y="${yc+4}" text-anchor="middle" font-size="11">Σ</text>`;
  let wires='', nodes='';
  rows.forEach((s,r) => {
    const yc=y(r);
    nodes += `<text x="${LW-12}" y="${yc+4}" text-anchor="end" font-size="11">${esc(s.label)}</text>`;
    wires += `<line class="wire" x1="${LW-6}" x2="${sEnd}" y1="${yc}" y2="${yc}"/>`;
    const fl = flags.stem[s.label];
    s.chain.forEach((t,k) => nodes += box(t, LW + k*ST, yc, fl && fl.has(t)));
  });
  groups.forEach((o,gi) => {
    const by = busYs[gi];
    o.idx.forEach(r => wires += curve(sEnd, y(r), busX-9, by));
    wires += `<line class="wire" x1="${busX+9}" x2="${bEnd}" y1="${by}" y2="${by}"/>` + curve(bEnd, by, mX-9, mY);
    nodes += sum(busX, by) + `<text class="busl" x="${bStart}" y="${by-BH/2-3}">${esc(o.g.name.replace(/_/g,' '))}</text>`;
    const fl = flags.bus[norm(o.g.name)];
    o.g.chain.forEach((t,k) => nodes += box(t, bStart + k*ST, by, fl && fl.has(t)));
  });
  direct.forEach(r => wires += curve(sEnd, y(r), mX-9, mY));
  wires += `<line class="wire" x1="${mX+9}" x2="${outX-7}" y1="${mY}" y2="${mY}"/>`;
  nodes += sum(mX, mY) + `<text class="busl" x="${mStart}" y="${mY-BH/2-3}">full mix</text>`;
  plan.master.forEach((t,k) => nodes += box(t, mStart + k*ST, mY, flags.master.has(t)));
  nodes += `<circle cx="${outX}" cy="${mY}" r="7" style="fill:var(--accent)"/><text class="busl" x="${outX}" y="${mY+20}" text-anchor="middle">out</text>`;
  let shade='';
  if (stages){
    let bi=0; const band=(x1,x2,lab)=>`<g class="band" data-stage="${bi++}" tabindex="0" role="button" aria-label="Show ${lab}"><rect x="${x1}" y="2" width="${x2-x1}" height="${H-4}" rx="8"/><text class="busl bandl" x="${x1+8}" y="16">${lab}</text></g>`;
    shade = band(4, sEnd+12, '1. Track level') + band(busX-16, bEnd+14, '2. Subgroup level') + band(mX-16, W-4, '3. Full-mix level');
  }
  return `<svg viewBox="0 0 ${W} ${H}" width="${W}" style="width:100%;min-width:${Math.min(W,900)}px;max-width:${Math.round(W*1.35)}px" role="${stages?"group":"img"}" aria-label="MixAgent console: ${rows.length} tracks, ${plan.groups.length} subgroups, full-mix effects ${plan.master.join(', ')}">${shade}${wires}${nodes}</svg>`;
}
function paramTable(vals){
  const body = vals.map(v => `<tr class="fxrow"><td colspan="3">${chip(v.tool)}</td></tr>` +
    (v.rows.length ? v.rows.map(r => `<tr><td>${esc(r[0])}</td><td class="v">${esc(r[1])}</td><td class="v after">${esc(r[2])}</td></tr>`).join('')
      : '<tr><td colspan="3" class="same">unchanged in the final mix</td></tr>')).join('');
  return `<div class="ptab"><table><thead><tr><th scope="col">Setting</th><th scope="col">First value</th><th scope="col">Corrected</th></tr></thead><tbody>${body}</tbody></table></div>`;
}
function jsonText(o){
  // two-space JSON with number arrays kept on one line
  return JSON.stringify(o, null, 2).replace(/\[\s+([-\d.,\s]+?)\s+\]/g, (m, inner) => '[' + inner.replace(/\s+/g, ' ') + ']');
}
// one critic card: A1 (what it hears), A2 (per effect), A3 (JSON on demand)
function criticCard(log, C, i){
  const r = log[i], c = C[i] || {};
  const name = r.element.replace(/ bus$/,' subgroup').replace(/^master$/i,'Full mix');
  const a2 = (c.a2||[]).map(([t,txt]) => `<div class="cc-fx"><span class="cc-fxn" style="color:var(--fx-${t},var(--ink));border-color:color-mix(in srgb,var(--fx-${t},var(--ink-3)) 45%,transparent)">${esc(FX[t]?FX[t][1]:t)}</span><p>${esc(txt)}</p></div>`).join('');
  const a1 = c.a1 ? `<section><h5><span class="atag">A1</span>What it hears</h5><p class="cc-a1">${esc(c.a1)}</p></section>`
    : (r.diag.length ? `<section><h5><span class="atag muted">A3</span>Diagnosis</h5><p class="cc-a1">${esc(r.diag.join(' '))}</p></section>` : '');
  return `<article class="cc"><header><span class="cc-lvl">${{stem:'Track',bus:'Subgroup',master:'Full mix'}[r.level]||r.level}</span><h4>${esc(name)}</h4></header>
    ${a1}${a2 ? `<section><h5><span class="atag">A2</span>What to change, effect by effect</h5>${a2}</section>` : ''}
    ${c.a3 ? `<section><button type="button" class="jsonbtn" aria-expanded="false"><span class="atag">A3</span>View parameter output (JSON)</button><pre class="json" hidden><code>${esc(jsonText(c.a3))}</code></pre></section>` : ''}</article>`;
}
// picker rows for every corrected element: tracks by subgroup, then subgroups, then the full mix
function criticPicks(ex, log){
  const used = new Set(), at = (lvl, f) => { const i = log.findIndex((r,k) => !used.has(k) && r.level === lvl && f(r)); if (i >= 0) used.add(i); return i; };
  const low = x => String(x).toLowerCase().replace(/_/g,' ').replace(/ bus$/,'').trim();
  const rows = ex.plan.groups.map(g => [g.name.replace(/_/g,' '), g.stems.map(st => [st.label, at('stem', r => r.element === st.label)]).filter(p => p[1] >= 0)]);
  const direct = (ex.plan.direct||[]).map(st => [st.label, at('stem', r => r.element === st.label)]).filter(p => p[1] >= 0);
  if (direct.length) rows.push(['direct', direct]);
  rows.push(['subgroups', ex.plan.groups.map(g => [g.name.replace(/_/g,' '), at('bus', r => low(r.element) === low(g.name))]).filter(p => p[1] >= 0)]);
  rows.push(['full mix', log.map((r,i) => r.level === 'master' && !used.has(i) ? ['Full mix', i] : null).filter(Boolean)]);
  const rest = log.map((r,i) => used.has(i) || r.level === 'master' ? null : [r.element, i]).filter(Boolean);
  if (rest.length) rows.push(['other', rest]);
  return rows.filter(r => r[1].length);
}
const CRIT = {};
function logPanel(log, key){
  const C = (typeof CRITIC !== 'undefined' && CRITIC[key]) || [];
  const hasF12 = C.some(c => c.a1 || c.a2);
  const intro = hasF12
    ? (key.startsWith('medleydb/')
      ? '<p class="lognote"><b>A1</b> perceptual feedback and <b>A2</b> effect-wise advice come from MixAgent runs with the A1 and A2 critics on the same clip. <b>A3</b> is the parameter output of the A3 critic that made this mix.</p>'
      : '<p class="lognote"><b>A1</b> perceptual feedback and <b>A2</b> effect-wise advice come from the A1 and A2 critics, given the same audio and the same element that the A3 critic corrected in this mix. <b>A3</b> is the A3 critic\u2019s parameter output.</p>')
    : '<p class="lognote">A1 and A2 critic runs exist for the MedleyDB clips only. Shown here: the A3 critic’s diagnosis and its parameter output (<b>A3</b>).</p>';
  const ex = CLIPS.find(c => c.ds+'/'+c.ex.id === key).ex, rows = criticPicks(ex, log);
  CRIT[key] = {log, C};
  const first = rows.length ? rows[0][1][0][1] : -1;
  return intro + `<div class="critbox" data-key="${esc(key)}"><div class="wl-pick">${rows.map(([g,items]) =>
    `<div class="wl-row"><span class="wl-g">${esc(g)}</span>${items.map(([l,i]) => `<button type="button" data-i="${i}" aria-pressed="${i===first}">${esc(l)}</button>`).join('')}</div>`).join('')}</div>
    <div class="wl-card">${first >= 0 ? criticCard(log, C, first) : ''}</div></div>`;
}
document.addEventListener('click', e => {
  const p = e.target.closest('.critbox .wl-pick button'); if (!p) return;
  const box = p.closest('.critbox'), k = CRIT[box.dataset.key], i = +p.dataset.i;
  box.querySelector('.wl-card').innerHTML = criticCard(k.log, k.C, i);
  box.querySelectorAll('.wl-pick button').forEach(b => b.setAttribute('aria-pressed', String(b === p)));
});
document.addEventListener('click', e => {
  const b = e.target.closest('.jsonbtn'); if (!b) return;
  const pre = b.nextElementSibling, open = pre.hidden;
  pre.hidden = !open; b.setAttribute('aria-expanded', String(open));
  b.lastChild.textContent = (open ? 'Hide' : 'View') + ' parameter output (JSON)';
});

function renderMixingRecord(content, ds, ex, note){
  content.innerHTML = `<header class="record-header"><p class="record-label">${esc(DSNAME[ds])}, mixing record</p><h2>${esc(note.label)}</h2></header>
  <div class="record-tabs" role="tablist" aria-label="Mixing record sections">
    <button type="button" role="tab" id="record-tab-corrections" aria-controls="record-corrections" aria-selected="true">Correction log <span>${ex.log.length}</span></button>
    <button type="button" role="tab" id="record-tab-plan" aria-controls="record-plan" aria-selected="false" tabindex="-1">Mixing plan</button>
    <button type="button" role="tab" id="record-tab-tracks" aria-controls="record-tracks" aria-selected="false" tabindex="-1">Input tracks <span>${ex.stems.length}</span></button>
  </div>
  <div role="tabpanel" id="record-corrections" aria-labelledby="record-tab-corrections">${logPanel(ex.log, ds+'/'+ex.id)}</div>
  <div role="tabpanel" id="record-plan" aria-labelledby="record-tab-plan" hidden><div class="console">${consoleSVG(ex.plan)}</div><p class="console-help">Follow each track’s effects into its subgroup and the final mix.</p></div>
  <div role="tabpanel" id="record-tracks" aria-labelledby="record-tab-tracks" hidden>${stemsPanel(ex)}</div>`;
  const tabs = [...content.querySelectorAll('[role="tab"]')];
  const select = tab => {
    content.querySelectorAll('audio').forEach(a => a.pause());
    tabs.forEach(t => {
      const active = t === tab;
      t.setAttribute('aria-selected', String(active)); t.tabIndex = active ? 0 : -1;
      content.querySelector('#'+t.getAttribute('aria-controls')).hidden = !active;
    });
  };
  tabs.forEach((tab,i) => {
    tab.onclick = () => select(tab);
    tab.onkeydown = e => {
      if (!['ArrowLeft','ArrowRight','Home','End'].includes(e.key)) return;
      e.preventDefault();
      const next = e.key === 'Home' ? 0 : e.key === 'End' ? tabs.length-1 : (i+(e.key==='ArrowRight'?1:-1)+tabs.length)%tabs.length;
      select(tabs[next]); tabs[next].focus();
    };
  });
}
// generic tablist: arrow keys, Home/End, panels via aria-controls
function tabset(tabs, onSelect){
  const select = (i, focus) => {
    document.querySelectorAll('audio').forEach(a => a.pause());
    tabs.forEach((t,k) => {
      t.setAttribute('aria-selected', String(k===i)); t.tabIndex = k===i ? 0 : -1;
      document.getElementById(t.getAttribute('aria-controls')).hidden = k!==i;
    });
    if (focus) tabs[i].focus();
    if (onSelect) onSelect(i);
  };
  tabs.forEach((t,i) => {
    t.onclick = () => select(i);
    t.onkeydown = e => {
      if (!['ArrowLeft','ArrowRight','Home','End'].includes(e.key)) return;
      e.preventDefault();
      select(e.key==='Home' ? 0 : e.key==='End' ? tabs.length-1 : (i+(e.key==='ArrowRight'?1:-1)+tabs.length)%tabs.length, true);
    };
  });
  return select;
}
// audio examples: original demo layout, one card per mix
const GROUPS = [["medleydb","MedleyDB"],["albumdb","AlbumDB"],["real","Listening test"]];
const CLIPS = GROUPS.flatMap(([ds,dn]) => DATA.datasets[ds].map((ex,k) => ({ds,dn,ex,note:Object.assign({label:`Sample ${k+1}`}, NOTES[ds][ex.id])})));
const $ = id => document.getElementById(id);
let current = 0, recordOpen = true;
function cell(ex, c, ds){
  const real = ds === 'real', C = real ? COND_REAL : COND, tgt = real ? 'reference' : 'pro';
  const cls = c === 'mixagent' ? ' ours' : c === tgt ? ' target' : '';
  const tag = c === 'mixagent' ? '<em>ours</em>' : c === tgt ? '<em>target</em>' : (real && c === 'pro') ? '<em class="anchor">anchor</em>' : STEM4.has(c) ? '<em class="anchor">4-stem based</em>' : '';
  const rating = (ex.ratings && ex.ratings[c] != null) ? `<div class="rating"><span>Listener rating</span><b>${ex.ratings[c].toFixed(1)}</b></div>` : '';
  return `<div class="cell${cls}" data-cond="${c}"><div class="ch">${C[c][0]}${tag}</div>${player(ex.cells[c].mp3)}${rating}${meter(ex.cells[c].feat, c===tgt, real ? 'style reference' : 'professional mix')}</div>`;
}
function renderDeck(){
  const c = CLIPS[current], ex = c.ex;
  const real = c.ds === 'real';
  const row = list => list.filter(k => ex.cells[k]).map(k => cell(ex,k,c.ds)).join('');
  const prior = ROW_PRIOR.filter(k => ex.cells[k]);
  $('deck').innerHTML = `<div class="exhead"><h3>${esc(c.dn)}, ${esc(c.note.label)}</h3><p class="tags">${esc(c.note.tags.replace(/ · /g,', '))}</p></div>
    <div class="exnote"><b><span aria-hidden="true">🎧</span> What to listen for:</b> ${esc(c.note.note)}</div>
    <div class="cellrow"><h4>${real ? "Style reference and the song's own mix" : 'Style reference and professional mix'}</h4><div class="cells">${row(ROW_TARGETS)}</div></div>
    <div class="cellrow"><h4>Mixes from the input tracks</h4><div class="cells">${row(ROW_SYSTEMS)}</div></div>
    ${prior.length ? `<div class="cellrow"><h4>Prior neural mixing systems</h4><div class="cells">${row(ROW_PRIOR)}</div></div>` : ''}
    <button type="button" class="btn" id="record-open" aria-controls="mixing-record"></button>`;
  const btn = $('record-open'), r = $('mixing-record');
  const show = open => {
    recordOpen = open;
    if (open) renderMixingRecord(r, c.ds, ex, c.note);
    r.hidden = !open; btn.setAttribute('aria-expanded', String(open));
    btn.textContent = (open ? 'Hide' : 'Show') + " this clip's corrections, mixing plan, and input tracks";
  };
  show(recordOpen);
  btn.onclick = () => { document.querySelectorAll('audio').forEach(a => a.pause()); show(!recordOpen); if (recordOpen) r.scrollIntoView({block:'start', behavior:'smooth'}); };
  document.querySelectorAll('.clips button').forEach((b,i) => b.setAttribute('aria-current', i===current ? 'true':'false'));
}
const pills = ds => CLIPS.map((c,i)=>c.ds===ds?`<button type="button" data-i="${i}">${esc(c.note.label)}</button>`:'').join('');
const groupRow = ([ds,dn]) => `<div class="clips"><h4>${dn}</h4><div class="pills">${pills(ds)}</div></div>`;
const SETS = [
  ["same", "Same-song reference", "The style reference is another part of the same song, and the professional mix of the clip is the target.", GROUPS.filter(g => g[0] !== 'real')],
  ["diff", "Different-song reference", "Songs from the listening test. The style reference comes from a different song of the same genre, so there is no exact target mix. The charts compare each mix with the style reference, and the song's own mix is shown only as an anchor.", GROUPS.filter(g => g[0] === 'real')]
];
$('clip-list').innerHTML = `<div class="record-tabs settabs" role="tablist" aria-label="Reference type">${SETS.map(([k,t],i) =>
    `<button type="button" role="tab" id="settab-${k}" aria-controls="setpanel-${k}" aria-selected="${i===0}" tabindex="${i===0?0:-1}">${t}</button>`).join('')}</div>
  ${SETS.map(([k,,d,gs],i) => `<div class="setpanel" role="tabpanel" id="setpanel-${k}" aria-labelledby="settab-${k}"${i?' hidden':''}><p>${d}</p>${gs.map(groupRow).join('')}</div>`).join('')}`;
const pick = i => { document.querySelectorAll('audio').forEach(a => a.pause()); current = i; renderDeck(); };
document.querySelectorAll('.clips button').forEach(b => b.onclick = () => pick(+b.dataset.i));
tabset([...document.querySelectorAll('.settabs [role="tab"]')], i => {
  const want = SETS[i][3].map(g => g[0]);
  if (!want.includes(CLIPS[current].ds)) pick(CLIPS.findIndex(c => want.includes(c.ds)));
});
renderDeck();
const howto = {bands:[-7.8,1.2,-3.5,3.3], width:0.05, width_t:0.26};
$('howto-chart').innerHTML = meter(howto,false,'professional mix');

// walkthrough: one clip through the three stages, planner settings vs corrected
(function(){
  const w = DATA.walk, A = w.audio, cn = w.counts || {};
  const ex = DATA.datasets[w.dataset].find(e => e.id === w.id);
  document.getElementById('walk-console').innerHTML = consoleSVG(ex.plan, true);

  const ab = (a, b, la, lb) => `<div class="ab"><div><label>${la||"Planner's first settings"}</label>${player(A[a])}</div><div class="after"><label>${lb||'After correction'}</label>${player(A[b])}</div></div>`;
  const key = {brass:'A_g0_brass', bass:'A_g1_bass', piano:'A_g2_piano', drums:'A_g3_drums'};
  const head = `<thead><tr><th scope="col">Subgroup</th><th scope="col">Qwen plan only</th><th scope="col" class="after">MixAgent</th></tr></thead>`;
  const trow = (name, sub, a, b) => `<tr><th scope="row" data-g="${esc(name.replace(/ subgroup$/,''))}"><b>${name}</b><span>${sub}</span></th><td>${player(A[a])}</td><td class="after">${player(A[b])}</td></tr>`;
  const sub = g => g.stems.map(s=>esc(s.label)).join(', ');
  const level = (n, title, out, flow, body, last) => `<section class="lvl-block" role="tabpanel" id="stage-panel-${n}" aria-labelledby="stage-tab-${n}"${n===1?'':' hidden'}>
      <div class="lvl-head"><span class="lvl-n">${n}</span><div><h3>${title}</h3><p class="flow">${flow}</p><p>${out}</p></div></div>
      <div class="walkgrid"><div>${body}</div><aside class="walklog" data-level="${['stem','bus','master'][n-1]}" aria-label="What the critics said at this level"></aside></div></section>`;
  // track level: every track after its own effects, before the subgroup sum
  const tk = l => String(ex.stems.findIndex(s => s.label === l)).padStart(2,'0');
  const t1 = `<div class="walktable tracks"><table><thead><tr><th scope="col">Track</th><th scope="col">Qwen plan only</th><th scope="col" class="after">MixAgent</th></tr></thead><tbody>${ex.plan.groups.map(g =>
    `<tr class="grow"><th colspan="3" scope="colgroup" data-g="${esc(g.name)}">${esc(g.name)}</th></tr>` + g.stems.map(st =>
      `<tr><th scope="row" data-t="${esc(st.label)}"><b>${esc(st.label)}</b></th><td>${player(`audio/walk/tracks/TQ_${tk(st.label)}.mp3`)}</td><td class="after">${player(`audio/walk/tracks/TM_${tk(st.label)}.mp3`)}</td></tr>`).join('')).join('')}</tbody></table></div>`;
  const t2 = `<div class="walktable"><table>${head}<tbody>${ex.plan.groups.map(g => trow(esc(g.name)+' subgroup', 'effects applied to the combined tracks', 'Q_S_'+g.name, 'S_'+g.name+'_2')).join('')}
      ${trow('Summed mix', 'all four subgroups added together', 'Q_B', 'B_2')}</tbody></table></div>`;
  const t3 = `<div class="walktable"><table><thead><tr><th scope="col"></th><th scope="col">Qwen plan only</th><th scope="col" class="after">Final MixAgent mix</th><th scope="col">Professional mix</th></tr></thead>
      <tbody><tr><th scope="row" data-m="1"><b>Final mix</b><span>effects applied to the full mix</span></th><td>${player(A.Q_C)}</td><td class="after">${player(A.C_2)}</td><td>${player(A.pro)}</td></tr></tbody></table></div>`;
  document.getElementById('walk').innerHTML =
    level(1, 'Track level', `Effects are applied to each track separately. Each player is one track after its own effects, before the subgroup sum. The two versions of a track play at the same gain, so level changes stay audible. ${cn.stem||0} corrections.`,
      '<span>tracks</span>→<span>track effects</span>→<span>Σ per subgroup</span>', t1) +
    level(2, 'Subgroup level', `Effects are applied to each subgroup, such as brass or drums. The subgroups are then combined. ${cn.bus||0} corrections.`,
      '<span>track-level output</span>→<span>subgroup effects</span>→<span>Σ all subgroups = summed mix</span>', t2) +
    level(3, 'Full-mix level', `Effects are applied to the full mix, with all instruments together. ${cn.master||0} correction.`,
      '<span>summed mix</span>→<span>full-mix effects</span>→<span>final mix</span>', t3, true);
  const stages = ['Track level','Subgroup level','Full-mix level'];
  const nav = document.getElementById('stage-navigation');
  nav.innerHTML = stages.map((title,i) => `<button type="button" class="stage-tab" role="tab" id="stage-tab-${i+1}" aria-controls="stage-panel-${i+1}" aria-selected="${i===0}" tabindex="${i===0?0:-1}">${i+1}. ${title}</button>`).join('');
  const buttons = [...nav.querySelectorAll('button')];
  const bands = [...document.querySelectorAll('#walk-console .band')];
  const selectStage = index => {
    document.querySelectorAll('#walk audio').forEach(a => a.pause());
    bands.forEach((b,i) => b.classList.toggle('on', i===index));
    buttons.forEach((button,i) => {
      button.setAttribute('aria-selected',String(i===index)); button.tabIndex = i===index?0:-1;
      document.getElementById('stage-panel-'+(i+1)).hidden = i!==index;
    });
  };
  bands.forEach((b,i) => {
    const go = () => { selectStage(i); document.getElementById('stage-navigation').scrollIntoView({block:'nearest',behavior:'smooth'}); };
    b.onclick = go;
    b.onkeydown = e => { if (e.key==='Enter' || e.key===' ') { e.preventDefault(); go(); } };
  });
  // what the critics said: one element at a time, picked by name (grouped by subgroup at the track level)
  const C = (typeof CRITIC !== 'undefined' && CRITIC[w.dataset+'/'+w.id]) || [];
  const idx = {}; ex.log.forEach((r,i) => idx[r.level+':'+r.element] = i);
  const card = i => criticCard(ex.log, C, i);
  document.querySelectorAll('.walklog').forEach(box => {
    const lv = box.dataset.level;
    let picks;
    if (lv === 'stem') picks = ex.plan.groups.map(g => [g.name, g.stems.map(s => s.label).filter(l => idx['stem:'+l] != null).map(l => [l, idx['stem:'+l]])]);
    else if (lv === 'bus') picks = [[null, ex.plan.groups.map(g => [g.name.replace(/_/g,' '), ex.log.findIndex(r => r.level==='bus' && r.element.toLowerCase().replace(/ bus$/,'') === g.name.replace(/_/g,' '))]).filter(p => p[1] >= 0)]];
    else picks = [[null, ex.log.map((r,i) => r.level==='master' ? ['Full mix', i] : null).filter(Boolean)]];
    const all = picks.flatMap(p => p[1]);
    box.innerHTML = `<h4 class="wl-h">What the critics said</h4>${all.length > 1 ? `<div class="wl-pick">${picks.map(([g,items]) =>
      `<div class="wl-row">${g ? `<span class="wl-g">${esc(g)}</span>` : ''}${items.map(([l,i]) => `<button type="button" data-i="${i}">${esc(l)}</button>`).join('')}</div>`).join('')}</div>` : ''}<div class="wl-card"></div>`;
    // the table row (left) that belongs to a card: track at the track level, subgroup at the subgroup level, the final mix at the top
    const busOf = g => (picks[0][1].find(([l]) => l === g.replace(/_/g,' ')) || [])[1];
    const rowOf = th => th.dataset.t != null ? idx['stem:'+th.dataset.t] : th.dataset.m ? all[0] && all[0][1] : lv === 'bus' ? busOf(th.dataset.g) : undefined;
    const rows = [...box.closest('.lvl-block').querySelectorAll('tbody th[data-t], tbody th[data-m]' + (lv === 'bus' ? ', tbody th[data-g]' : ''))];
    const show = i => { box.querySelector('.wl-card').innerHTML = card(i); box.querySelectorAll('.wl-pick button').forEach(b => b.setAttribute('aria-pressed', String(+b.dataset.i === i)));
      rows.forEach(th => th.parentElement.classList.toggle('sel', rowOf(th) === i)); };
    box.querySelectorAll('.wl-pick button').forEach(b => b.onclick = () => show(+b.dataset.i));
    // clicking a track or subgroup name in the table opens its card
    box.closest('.lvl-block').querySelectorAll('th[data-t]').forEach(th => th.onclick = () => { const i = idx['stem:'+th.dataset.t]; if (i != null) show(i); });
    box.closest('.lvl-block').querySelectorAll('th[data-g]').forEach(th => th.onclick = () => {
      const i = lv === 'bus' ? busOf(th.dataset.g) : ((picks.find(([g]) => g === th.dataset.g) || picks[0])[1][0] || [])[1];
      if (i != null) show(i);
    });
    box.closest('.lvl-block').querySelectorAll('th[data-m]').forEach(th => th.onclick = () => { const i = rowOf(th); if (i != null) show(i); });
    if (all.length) show(all[0][1]);
  });
  selectStage(0);
  buttons.forEach((button,i) => {
    button.onclick = () => selectStage(i);
    button.onkeydown = e => {
      if (!['ArrowLeft','ArrowRight','Home','End'].includes(e.key)) return;
      e.preventDefault();
      const next = e.key === 'Home' ? 0 : e.key === 'End' ? 2 : (i+(e.key==='ArrowRight'?1:2))%3;
      selectStage(next); buttons[next].focus();
    };
  });

})();
// page sections: one tab at a time, kept in the URL hash
(function(){
  const tabs = [...document.querySelectorAll('.sectabs [role="tab"]')];
  const ids = tabs.map(t => t.getAttribute('aria-controls'));
  const select = tabset(tabs, i => { if (location.hash.slice(1) !== ids[i]) history.replaceState(null, '', '#'+ids[i]); });
  const fromHash = () => { const i = ids.indexOf(location.hash.slice(1)); if (i >= 0) select(i); return i >= 0; };
  window.addEventListener('hashchange', () => { if (fromHash()) document.querySelector('.sectabs').scrollIntoView({block:'start'}); });
  if (!fromHash()) select(0);
  document.querySelectorAll('a[href^="#"]').forEach(l => { const i = ids.indexOf(l.getAttribute('href').slice(1)); if (i >= 0) l.onclick = e => { e.preventDefault(); select(i); document.querySelector('.sectabs').scrollIntoView({block:'start', behavior:'smooth'}); }; });
  const helps = [...document.querySelectorAll('.helpbtn')];
  helps.forEach(b => document.getElementById(b.getAttribute('aria-controls')).hidden = true);
  helps.forEach(b => b.onclick = () => {
    const open = b.getAttribute('aria-expanded') !== 'true';
    helps.forEach(o => { const on = open && o === b; o.setAttribute('aria-expanded', String(on)); document.getElementById(o.getAttribute('aria-controls')).hidden = !on; });
  });
})();
// effects glossary
const fxBox = t => `<svg class="fxbox" viewBox="0 0 68 22" width="68" height="22" aria-hidden="true"><rect x="1" y="1" width="66" height="20" rx="4" style="fill:color-mix(in srgb,var(--fx-${t}) 14%,var(--panel));stroke:var(--fx-${t})" stroke-width="1.3"/><svg x="4" y="6" width="15" height="10" viewBox="0 0 44 30" style="color:var(--fx-${t})" overflow="visible">${ICON[t]}</svg><text x="21" y="14.5" font-size="9.5" font-weight="500" style="fill:var(--ink);font-family:var(--f-body)">${esc(FX[t][0])}</text></svg>`;
document.getElementById('fxgrid').innerHTML = Object.entries(FX).map(([k,v]) => `<div class="fx">
  <div class="fxtitle">${fxBox(k)}<b>${v[1]}</b></div><span>${v[2]}</span></div>`).join('');
// overview video: show a placeholder until assets/overview.mp4 exists
const vframe = document.querySelector('.vframe');
if (vframe) {
  const v = vframe.querySelector('video'), src = v.querySelector('source');
  const empty = on => { vframe.classList.toggle('empty', on); vframe.querySelector('.vsoon').hidden = !on; };
  // only a failed load shows the placeholder (networkState reads NO_SOURCE briefly while the source is being picked)
  src.addEventListener('error', () => empty(true));
  v.addEventListener('loadeddata', () => empty(false));
  window.addEventListener('load', () => { if (!v.currentSrc && v.networkState === HTMLMediaElement.NETWORK_NO_SOURCE) empty(true); });
  // autoplay (muted, as browsers require) while the video is on screen, unless the viewer paused it
  let userPaused = false, auto = false;
  v.addEventListener('pause', () => { if (!auto && !v.ended) userPaused = true; auto = false; });
  v.addEventListener('play', () => { userPaused = false; });
  new IntersectionObserver(([e]) => {
    if (e.isIntersecting) { if (!userPaused && v.paused && !v.ended) v.play().catch(() => {}); }
    else if (!v.paused) { auto = true; v.pause(); }
  }, {threshold: 0.25}).observe(v);
}
// one player at a time; switching between mixes of the same clip keeps the position
document.addEventListener('play', e => {
  const t = e.target, sameClip = a => a.closest && a.closest('.cell') && a.closest('.cell').dataset.cond !== 'reference';
  document.querySelectorAll('audio, video').forEach(a => {
    if (a === t) return;
    if (!a.paused && sameClip(a) && sameClip(t)) t.currentTime = a.currentTime;
    a.pause();
  });
}, true);
})();
