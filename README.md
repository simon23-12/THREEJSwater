# Stormy Ocean — FFT

Ein Ozean in three.js (WebGL2), gebaut auf einer echten spektralen Wellen­simulation
statt auf überlagerten Sinus­wellen.

```bash
npm install
npm run dev      # http://localhost:5173
```

## Wie es funktioniert

**Wellen — inverse FFT auf der GPU.**
Aus einem **JONSWAP**-Spektrum (fetch-begrenzte Windsee) mit **TMA**-Tiefen­korrektur und
**Hasselmann**-Richtungs­verteilung wird einmalig das gefrorene Spektrum `h0(k)` erzeugt
(`src/ocean/spectrum.js`, CPU, deterministisch geseedet). Pro Frame läuft dann in
`src/ocean/OceanSim.js`:

1. Spektrum auf Zeit `t` entwickeln — `h(k,t) = h0(k)·e^{iωt} + conj(h0(−k))·e^{−iωt}`, Dispersion `ω = √(gk)`
2. 2-D inverse FFT über eine vorberechnete Butterfly-Tabelle, 8 horizontale + 8 vertikale Stufen bei 256²
3. Entmischen zu einer Displacement-Map `(Dx, Höhe, Dz)`
4. Schaum aus der **Jacobi-Determinante** der Verschiebung — wo sich die Oberfläche
   faltet (`J < 0`), bricht sie; der Schaum wird akkumuliert und klingt zeitlich ab

Zwei reelle Felder teilen sich eine komplexe Transformation, und beide Kanäle einer RGBA-Textur
werden gleichzeitig geschmetterlingt — das halbiert die Anzahl der Passes.

Drei **Kaskaden** (Kachelgrößen 360 m / 51 m / 11 m) teilen sich das Spektrum nach Wellen­zahl auf,
damit sich Dünung und Kräusel­wellen nicht doppeln und keine sichtbare Kachelung entsteht.

**Oberfläche.** Ein radiales Clipmap-Gitter (400 Ringe × 512 Segmente, exponentiell wachsend
bis 17 km) wird jeden Frame auf die Kamera zentriert — ein Draw-Call für einen praktisch
unendlichen Ozean. Normalen und die Faltung kommen per finiter Differenzen aus der
Displacement-Map, das Sample-Fenster wächst mit der Distanz und wirkt so als Tiefpass gegen
Specular-Flimmern.

**Licht.** Der Himmel ist prozedural (Gradient + Horizont­dunst + Wolken­decke aus domain-warped
fbm) und wird pro Frame in eine Equirect-Environment-Map gebacken. Dieselbe Textur liefert
Hintergrund, Wasser­reflexion **und** Dunstfarbe — dadurch passt am Horizont alles exakt zusammen.
Dazu Fresnel, GGX-Sonnenglitzer, Subsurface-Streuung auf den Wellen­rücken, Bloom und ein
ACES-Filmic-Grade.

## Steuerung

| | |
|---|---|
| Maus ziehen | Umsehen |
| Scrollen | Brennweite |
| W A S D / Q E | Bewegen |
| Shift | schneller |
| **H** | Regler ein-/ausblenden |

Über die Regler lässt sich der Seegang live ändern: Windstärke und -richtung, Fetch,
Steilheit, Schaum, Dunst, Bewölkung, Belichtung. Wind/Fetch bauen `h0` neu auf (kurzer Hänger),
alles andere greift sofort.

`window.__ocean` gibt in der Konsole Zugriff auf `camera`, `sim`, `sky` und `params`.

## Technische Voraussetzungen

WebGL2 mit `EXT_color_buffer_float`. `OES_texture_float_linear` wird genutzt, wenn vorhanden,
sonst fallen die gesampelten Maps automatisch auf Half-Float zurück.
