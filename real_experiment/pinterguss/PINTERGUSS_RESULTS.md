# Pinterguss-Messstudie — Ergebnisse & Paper-Entscheidung (Stand 2026-08-26)

Destillat für die Entscheidung „fließt Pinterguss ins (Haupt-)Paper ein?"
(Nutzerwunsch 2026-08-26). Quellen: `final_metrics_pinterguss_20260822.csv`,
`pinterguss_slices.png`, Supplementary `sec:supp_pinterguss` (fertig
geschrieben & kompiliert).

## Was gemessen wurde

Zweites Realobjekt auf derselben Roboter-CT-Bank wie die Kamera: Aluminium-
Guss („Pinterguss", hakenförmig), 1095-View-Zirkularreferenz (quantitative
FDK, 768³ @ 0.2777 mm), Planungsprior = Pseudo-Prescan (jede 9. Projektion,
122 Views — Planung sieht die Referenz nie). Beide geplanten Band-Arme
(bundle, all3 = coverage+VCL+bundle) wurden von THD bei k ∈ {50, 100, 400}
physisch gemessen (je eine View fehlend: 49/99/399), dazu Zirkular-Subsets
und Uniform-on-Band als Baselines. Alles im Referenzrahmen rekonstruiert
(Geometrie-Registrierung, ASD-POCS, residuengewählte TV-Stärke) — Protokoll
identisch zur Kamerastudie. Besonderheit: Montagebasis ragt in JEDER View
unten aus dem FOV (Messeigenschaft, kein Rekonstruktionsartefakt); das
Gussteil selbst liegt vollständig im FOV.

## Kernzahlen (volle Tabelle: `final_metrics_pinterguss_20260822.csv`)

| Arm | k=50 PSNR/SSIM | k=100 PSNR/SSIM | k=400 PSNR/SSIM |
|---|---|---|---|
| circular (Referenz-Session) | 32.72 / 0.869 | 32.14 / 0.872 | 30.70 / 0.869 |
| uniform-on-band | 30.13 / 0.825 | 29.56 / 0.833 | 27.68 / 0.831 |
| all3 (geplant) | 29.62 / 0.814 | 28.94 / 0.817 | 27.80 / **0.831** |
| bundle (geplant) | 29.60 / 0.814 | 28.62 / 0.812 | 27.73 / 0.817 |

Lesart (Protokoll-Caveats wie Kamerastudie: PSNR/SSIM gegen die glatte
FDK-Referenz begünstigen niedrig-konvergierte kleine k; zirkular teilt die
Referenz-Session und ist doppelt bevorteilt; uniform-on-band poolt 1094
bereits gemessene Views — kein fairer Budgetvergleich):

1. **all3 ≥ bundle bei jedem k** (+0.02/+0.32/+0.07 dB; SSIM-Abstand wächst
   mit k auf +0.014). Das ist die erste Messevidenz FÜR den korrigierten
   VCL-Stack — diese Pläne entstanden (anders als die Kamera-Arme)
   vollständig unter korrigierter Quadratur (clip@512), komplettem
   VCL-Geometrie-VJP und evaluated-iterate Adam.
2. Bei k=400 schließt all3 zur Uniform-Baseline auf (PSNR +0.12 dB,
   SSIM gleichauf) — bei k=50/100 liegt uniform vorn.
3. Zirkular ≫ alle Band-Arme: erwarteter Selbstkonsistenz-Effekt
   (gleiche Session wie Referenz), Pipeline-Sanity-Check.

## Einschätzung zur Paper-Integration

**Empfehlung: im Supplementary belassen (ist dort fertig), im Haupttext ein
einzelner Verweissatz in sec:real.** Gründe:

- Der Mehrwert gegenüber der Kamerastudie ist der **Replikations- und
  Korrektur-Nachweis** (zweites Objekt, zweite Session, korrigierter
  Estimator-Stack, all3-Vorteil konsistent) — genau die Rolle einer
  Supplementary-Studie. Ein eigenständiges Hauptpaper-Ergebnis („geplant
  schlägt Baseline") liefert Pinterguss NICHT: uniform-on-band bleibt bei
  kleinen k vorn, mit denselben Referenz-Kopplungs-Caveats wie bei der
  Kamera.
- Das Hauptpaper hat mit der Kamera bereits ein vollständiges
  Realexperiment inkl. Abbildung; ein zweites kostet ~0.5–1 Seite gegen
  akuten Längendruck (Noise-Studie stand deshalb schon zur Debatte).
- Die FOV-Trunkierung der Montagebasis müsste im Haupttext erklärt werden
  (im Supp bereits sauber dokumentiert) — unnötige Angriffsfläche.

**Wenn doch Haupttext:** stärkste Form wäre EIN Satz + Verweis:
„A second measured study on an aluminium casting, acquired with the
corrected estimator stack, reproduces the protocol end-to-end and shows the
full composite matching or exceeding the bundle-only arm at every measured
budget (supplementary sec:supp_pinterguss)." — das habe ich bewusst noch
NICHT eingebaut; Entscheidung liegt bei dir.

## Offene Punkte

- Slice-Figur (`pinterguss_slices.png`) und Tabelle sind im Supp final;
  keine weiteren Läufe nötig.
- Falls Haupttext-Integration: Satz in `content/results.tex` (sec:real,
  Ende) + ggf. \cref auf die Supp-Sektion.
