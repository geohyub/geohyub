## Junhyub Kim · 김준협

Marine geophysical surveyor at GeoView, Busan.
I write the in-house software we use to QC, process and deliver survey data.

geohyub@gmail.com · [GeoView version index](https://github.com/geohyub/geoview-updates)

---

<img alt="SeismicQC headless CLI grading three SEG-Y lines, then QCHub importing the result files" src="assets/qc-pipeline.gif" width="100%">

<sub>A real run, recorded from its own output: the SeismicQC headless CLI grades three synthetic
UHR lines (one deliberately truncated, which the SEG-Y header check fails), writes
<code>geoview.qc-result/1</code> files, and QCHub imports them through its API into per-line status.
Re-importing a file updates the row instead of duplicating it.
Rendered by <a href="assets/make_pipeline_gif.py">assets/make_pipeline_gif.py</a>.</sub>

### GeoView

In-house tools for offshore survey work, from the vessel to the client delivery.
41 programs are in the release index; most are Python / PySide6 desktop apps,
some are web apps (Flask, FastAPI, Next.js) and a few are .NET. The source is private.

How they fit together: raw files are prepared and checked on board,
each QC app grades every line and writes a common result file,
QCHub and LineTracker collect those into project and line status,
and HandoverGenerator checks the delivery package before it leaves.

### By discipline

**Magnetics**
| Program | What it does |
|---|---|
| MagPrep | Preprocesses raw magnetometer files before QC |
| MagQC | Line QC: 4th-difference noise, contract (SoW) verdict per line, reports |
| MagProc | 8-stage processing pipeline with IGRF reference field removal |
| MagUXO | UXO target detection: Euler deconvolution plus a CNN ensemble |

**Multibeam and hydrography**
| Program | What it does |
|---|---|
| MBESQC | Multibeam QC: motion, offsets, SVP, density and coverage, DQR reports |
| CARISBoresight | Helper for boresight (patch test) calibration |
| CATZOCAssigner | Assigns CATZOC zone-of-confidence categories |
| SVPBatchViewer | Batch viewer for sound velocity profiles |
| MBESPerfExport | Exports multibeam performance metrics |
| FusionGrid | Fuses QC grids from several sources |

**Side-scan sonar**
| Program | What it does |
|---|---|
| SonarQC | Side-scan QC per line: altitude, track, position and noise stages |
| MosaicEye | Visual QC of sonar mosaics |
| SSSBTPicker | Side-scan bottom-tracking picker |

**Seismic (SBP / UHR)**
| Program | What it does |
|---|---|
| SeismicQC | SEG-Y QC: seafloor, navigation, SNR, bandwidth, dead traces, header checks |
| SeismicProcessor | Processing engine: geometry from navigation, velocity analysis, QC decks, daily reports |
| SGYCompare | Trace-level comparison of two SEG-Y files |
| segylint | SEG-Y rev1/rev2 spec linter with SARIF output for CI |
| Feathering | Streamer feathering analysis |

**Navigation, positioning and calibration**
| Program | What it does |
|---|---|
| NavQC | Navigation QC and P-format export (P1/90, P1/11, P2/94, P2/11) |
| P190NavConverter | Converts navigation to and from P1/90 |
| CoordConverter | 13 coordinate transformations: UTM, DMS, MGRS, P190, GeoJSON |
| OffsetManager | Vessel sensor offsets with 22 validation rules |
| CalibrationPro | Sensor calibration QC and certificates: GNSS, MAG altimeter, SVP, heading |

**Tide, current and metocean**
| Program | What it does |
|---|---|
| TidalWorks | Tidal and current harmonic analysis (UTide), reports |
| TideBedPy | Tidal correction for survey data |
| TideGuard | Validates tide data before it is applied |
| PValue | Weather-window (P-value) simulator for planning offshore operations |

**Geotechnical**
| Program | What it does |
|---|---|
| GTPrep · GTProc · GTQC | CPT / SPT / laboratory data: preparation, processing and QC |

**Delivery and reporting**
| Program | What it does |
|---|---|
| HandoverGenerator | Builds and validates the delivery package; blocks leaked local paths |
| DeliveryBinder | Binds delivery documents |
| DeliveryGuard | Checks delivered PDFs for redaction and watermark problems |
| ReportDraft · DailyReport · DocTemplates | Report drafting, daily reports, document templates |
| CommentTracker | Tracks client review comments and responses |

**Operations on board and in the office**
| Program | What it does |
|---|---|
| QCHub | Project QC hub: checklists, issues, imported QC results per line |
| LineTracker | Survey line status across the project |
| DataFerry | Distributes acquisition data from the vessel automatically |
| SyncChecker · NamingChecker | Verifies NAS copies and file naming rules |
| StorageManager · ArchiveManager | Storage audit (.NET) and project archiving |
| ControlTower · LaunchPad | Operations dashboard (Next.js) and launcher for the suite |
| ProjectManager · ProcessWorkflow | Project milestones and the process / task workflow |
| EquipMan | Equipment register with QR labels and certificates |
| WiringGen | Survey wiring diagram generator (DXF) |
| CalculatorPro | Survey calculators: seismic fold, resolution, NMO and more |
| PortDetector | Network and device monitor |

**Applied AI and research**
| Program | What it does |
|---|---|
| MarineSeismicAI | First-break picking ensemble with MC-Dropout uncertainty (PyTorch) |
| SeismicDenoising | DnCNN / Swin denoising for marine seismic |
| VelocityPicking | Machine-learning velocity picking |
| sbp-kit | SBP / UHR toolkit: robust SEG-Y I/O, self-supervised denoising |
| ReportLLM | Local-LLM report drafting with a guard on every number |
| QCMemory · VisionNarrator | QC case retrieval, and local vision-LLM QC comments |

### Stack

Python · PySide6 · Flask · FastAPI · NumPy / SciPy · segyio · pyproj · PyTorch ·
TypeScript · Next.js · Tauri · C# / .NET · GitHub Actions
