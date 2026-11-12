"""Diagnostic rules D01–D08 (diagnostics spec §3, IMP-D01–D08).

Each rule is a pure function of `DiagnosticInput` that implements its decision table top to
bottom. Missing data never becomes a verdict on its own: absence is only concluded from the
observations listed in `context`, and anything else is `UNKNOWN`, `INFO` or reduced
confidence.
"""

from __future__ import annotations

from dataclasses import dataclass

from gpu_system_inspector.diagnostics import context as ctx
from gpu_system_inspector.diagnostics.engine import DiagnosticInput, RegisteredRule
from gpu_system_inspector.diagnostics.evidence import (
    CONTAINER_TOOLKIT_URL,
    CUDA_COMPAT_URL,
    CUDA_ENV_VARS_URL,
    DCGM_URL,
    LINUX_INSTALL_URL,
    NVML_BINDING_URL,
    PYTORCH_SELECTOR_URL,
    WSL_GUIDE_URL,
    S,
    advice,
    finding,
    inference,
    major,
    not_applicable,
    observation,
    observed_field,
)
from gpu_system_inspector.models import (
    CollectionStatus,
    Confidence,
    DiagnosticFinding,
    Evidence,
    InspectionSnapshot,
    Observed,
    PlatformProfile,
    ReadinessReason,
    ReadinessStatus,
    Recommendation,
    RuntimeMetrics,
    Severity,
    ToolkitDiscovery,
)

_CONFIDENCE_RANK = {Confidence.HIGH: 0, Confidence.MEDIUM: 1, Confidence.LOW: 2}


def _driver_advice(s: InspectionSnapshot) -> Recommendation:
    if s.system.platform_profile is PlatformProfile.WSL2:
        return advice(
            "Install or update the NVIDIA driver on the Windows host (never inside WSL2).",
            "Under WSL2 the GPU is provided by the Windows host driver.",
            WSL_GUIDE_URL,
        )
    return advice(
        "Install or repair the NVIDIA driver for this GPU.",
        "CUDA applications need the NVIDIA kernel driver and its user-mode libraries.",
        LINUX_INSTALL_URL,
    )


# --------------------------------------------------------------------------------- D01

D01_TITLE = "PyTorch can't use CUDA"
D01_LIMITATION = (
    "Causes are inferred from the collected snapshot; other causes (for example corrupted "
    "installations or unusual library paths) are possible."
)
D01_5_LIMITATION = (
    "Same-major version differences are not reported as causes because of CUDA minor "
    "version compatibility; forward-compatibility packages are not detected."
)
D01_INIT_COMMAND = 'python -c "import torch; torch.cuda.init()"'
_PT = "/frameworks/pytorch"


@dataclass(frozen=True, slots=True)
class _Cause:
    code: str
    confidence: Confidence
    evidence: tuple[Evidence, ...]
    recommendation: Recommendation


def _d01_causes(s: InspectionSnapshot, *, gpu_observed: bool) -> list[_Cause]:
    pt, drv = s.frameworks.pytorch, s.driver
    causes: list[_Cause] = []
    if pt.cuda_build.is_collected and pt.cuda_build.value is False:
        causes.append(
            _Cause(
                "D01.1",
                Confidence.HIGH,
                (
                    observed_field("PyTorch cuda_build", pt.cuda_build, f"{_PT}/cuda_build"),
                    inference("D01.1: the installed PyTorch build has no CUDA support."),
                ),
                advice(
                    "Install a CUDA-enabled PyTorch build using the official selector.",
                    "A CPU-only build cannot use any GPU.",
                    PYTORCH_SELECTOR_URL,
                ),
            )
        )
    cvd = ctx.env(s, "CUDA_VISIBLE_DEVICES")
    if cvd is not None and cvd.is_collected and cvd.value is not None:
        first = cvd.value.split(",")[0]
        hidden = cvd.value.strip() == "" or (
            bool(s.gpu.devices) and not ctx.visible_entry_matches(first, s.gpu.devices)
        )
        if hidden:
            causes.append(
                _Cause(
                    "D01.2",
                    Confidence.HIGH,
                    (
                        observed_field(
                            "CUDA_VISIBLE_DEVICES", cvd, ctx.env_path("CUDA_VISIBLE_DEVICES")
                        ),
                        inference("D01.2: CUDA applications in this environment see no devices."),
                    ),
                    advice(
                        "Unset or correct CUDA_VISIBLE_DEVICES.",
                        "Its first entry decides whether CUDA applications see any device.",
                        CUDA_ENV_VARS_URL,
                    ),
                )
            )
    unbound = [
        (i, o)
        for i, o in enumerate(s.gpu.pci_observations.value or ())
        if o.is_nvidia and o.kernel_driver.value != ctx.NVIDIA_KERNEL_DRIVER
    ]
    if unbound and drv.nvml_version.status is CollectionStatus.NOT_DETECTED:
        i, o = unbound[0]
        causes.append(
            _Cause(
                "D01.3",
                Confidence.HIGH,
                (
                    observed_field(
                        f"PCI {o.address} kernel_driver",
                        o.kernel_driver,
                        f"/gpu/pci_observations/value/{i}/kernel_driver",
                    ),
                    observed_field("NVML", drv.nvml_version, "/driver/nvml_version"),
                    inference("D01.3: the NVIDIA kernel driver isn't controlling the GPU."),
                ),
                _driver_advice(s),
            )
        )
    library_missing = (
        drv.nvml_version.status is CollectionStatus.NOT_DETECTED
        and ctx.error_code(drv.nvml_version) == ctx.LIBRARY_NOT_FOUND
    ) or drv.cuda_driver_library.status is CollectionStatus.NOT_DETECTED
    if library_missing and gpu_observed:
        causes.append(
            _Cause(
                "D01.4",
                Confidence.MEDIUM,
                (
                    observed_field("NVML", drv.nvml_version, "/driver/nvml_version"),
                    observed_field(
                        "CUDA driver library",
                        drv.cuda_driver_library,
                        "/driver/cuda_driver_library",
                    ),
                    inference("D01.4: the CUDA driver library isn't loadable."),
                ),
                _driver_advice(s),
            )
        )
    driver_major = major(drv.driver_supported_cuda_version.value)
    build_major = major(pt.build_cuda_version.value)
    if driver_major is not None and build_major is not None and driver_major < build_major:
        causes.append(
            _Cause(
                "D01.5",
                Confidence.MEDIUM,
                (
                    observed_field(
                        "Driver-supported CUDA version",
                        drv.driver_supported_cuda_version,
                        "/driver/driver_supported_cuda_version",
                    ),
                    observed_field(
                        "PyTorch build CUDA version",
                        pt.build_cuda_version,
                        f"{_PT}/build_cuda_version",
                    ),
                    inference(
                        "D01.5: the driver likely doesn't support the CUDA major version "
                        "PyTorch was built with."
                    ),
                ),
                advice(
                    "Update the NVIDIA driver, or install a PyTorch build for an older CUDA "
                    "major version.",
                    "A driver supports CUDA up to the version it reports.",
                    CUDA_COMPAT_URL,
                ),
            )
        )
    if drv.nvml_version.status is CollectionStatus.PERMISSION_DENIED:
        causes.append(
            _Cause(
                "D01.6",
                Confidence.MEDIUM,
                (
                    observed_field("NVML", drv.nvml_version, "/driver/nvml_version"),
                    inference("D01.6: device access may be restricted for this user/container."),
                ),
                advice(
                    "Check the device permissions or the container GPU configuration.",
                    "NVML was denied access to the GPU.",
                    CONTAINER_TOOLKIT_URL,
                ),
            )
        )
    return sorted(causes, key=lambda c: (_CONFIDENCE_RANK[c.confidence], c.code))


def _d01_context(s: InspectionSnapshot) -> list[Evidence]:
    pt = s.frameworks.pytorch
    items = [
        observation(
            f"PyTorch reported a CUDA initialisation message: {message!r}",
            None,
            f"{_PT}/cuda_init_warnings/{i}",
        )
        for i, message in enumerate(pt.cuda_init_warnings)
    ]
    check = ctx.env(s, "PYTORCH_NVML_BASED_CUDA_CHECK")
    if check is not None and check.is_collected:
        items.append(
            observed_field(
                "PYTORCH_NVML_BASED_CUDA_CHECK (is_available() used NVML instead of the CUDA "
                "runtime)",
                check,
                ctx.env_path("PYTORCH_NVML_BASED_CUDA_CHECK"),
            )
        )
    return items


def d01(data: DiagnosticInput) -> list[DiagnosticFinding]:
    s = data.snapshot
    pt = s.frameworks.pytorch
    status_ev = observation(f"PyTorch section is {pt.status}", None, f"{_PT}/status")
    if pt.status in (CollectionStatus.NOT_INSTALLED, CollectionStatus.SKIPPED):
        why = (
            "skipped by --no-pytorch"
            if pt.status is CollectionStatus.SKIPPED
            else ("PyTorch is not installed in this Python environment")
        )
        return [not_applicable("D01", D01_TITLE, f"Not evaluated: {why}.", (status_ev,))]
    if pt.status is CollectionStatus.COLLECTION_FAILED:
        version_ev = observed_field("PyTorch version", pt.version, f"{_PT}/version")
        if ctx.error_code(pt.version) == "IMPORT_ERROR":
            return [
                finding(
                    "D01",
                    S.FAIL,
                    Severity.HIGH,
                    Confidence.HIGH,
                    D01_TITLE,
                    "PyTorch is installed but can't be imported, so it can't use CUDA.",
                    (version_ev,),
                    recommendations=(
                        advice(
                            "Repair the PyTorch installation in this environment.",
                            "The import error above is PyTorch's own message.",
                            PYTORCH_SELECTOR_URL,
                        ),
                    ),
                    limitations=(D01_LIMITATION,),
                )
            ]
        return [
            finding(
                "D01",
                S.UNKNOWN,
                Severity.MEDIUM,
                Confidence.NOT_APPLICABLE,
                D01_TITLE,
                "The PyTorch probe didn't complete, so CUDA usability is unknown.",
                (version_ev,),
                limitations=(D01_LIMITATION,),
            )
        ]
    if pt.rocm_build.is_collected and pt.rocm_build.value is True:
        return [
            not_applicable(
                "D01",
                D01_TITLE,
                "This is a ROCm build of PyTorch; ROCm is out of scope for this version.",
                (observed_field("PyTorch rocm_build", pt.rocm_build, f"{_PT}/rocm_build"),),
            )
        ]
    available = observed_field(
        "torch.cuda.is_available()", pt.cuda_available, f"{_PT}/cuda_available"
    )
    if not pt.cuda_available.is_collected:
        return [
            finding(
                "D01",
                S.UNKNOWN,
                Severity.MEDIUM,
                Confidence.NOT_APPLICABLE,
                D01_TITLE,
                "PyTorch's CUDA availability wasn't collected, so no conclusion was drawn.",
                (available,),
                limitations=(D01_LIMITATION,),
            )
        ]
    if pt.cuda_available.value is True:
        return [
            finding(
                "D01",
                S.PASS,
                Severity.NONE,
                Confidence.HIGH,
                D01_TITLE,
                "PyTorch reports that CUDA is available.",
                (available,),
                limitations=(D01_LIMITATION,),
            )
        ]
    gpu_observed = ctx.nvidia_observed(s)
    if not gpu_observed and ctx.nvidia_observed_absent(s):
        return [
            finding(
                "D01",
                S.INFO,
                Severity.NONE,
                Confidence.HIGH,
                D01_TITLE,
                "PyTorch reports CUDA unavailable, which is expected: no NVIDIA GPU was found.",
                (available, *_d01_context(s)),
                limitations=(D01_LIMITATION,),
            )
        ]
    causes = _d01_causes(s, gpu_observed=gpu_observed)
    evidence = [available]
    for cause in causes:
        evidence.extend(cause.evidence)
    evidence.extend(_d01_context(s))
    limitations = [D01_LIMITATION]
    if any(c.code == "D01.5" for c in causes):
        limitations.append(D01_5_LIMITATION)
    if causes:
        recommendations = [c.recommendation for c in causes]
        likely = ", ".join(c.code for c in causes)
        explanation = f"PyTorch reports CUDA unavailable. Likely causes, strongest first: {likely}."
    else:
        recommendations = [
            advice(
                f"Run {D01_INIT_COMMAND} to see PyTorch's full error.",
                "The collected evidence doesn't identify a cause.",
            )
        ]
        explanation = (
            "PyTorch reports CUDA unavailable; the cause couldn't be determined from the "
            "collected evidence."
        )
    confidence = causes[0].confidence if causes else Confidence.LOW
    if gpu_observed:
        status, severity = S.FAIL, Severity.HIGH
    else:
        status, severity = S.WARNING, Severity.MEDIUM
        confidence = Confidence.MEDIUM if confidence is Confidence.HIGH else confidence
        explanation += " Whether an NVIDIA GPU is present couldn't be determined."
    return [
        finding(
            "D01",
            status,
            severity,
            confidence,
            D01_TITLE,
            explanation,
            evidence,
            recommendations=recommendations,
            limitations=limitations,
        )
    ]


# --------------------------------------------------------------------------------- D02

D02_TITLE = "Free GPU memory vs requirement"
_GIB = 1 << 30


def _gib(n: int) -> str:
    return f"{n / _GIB:.1f} GiB"


def d02(data: DiagnosticInput) -> list[DiagnosticFinding]:
    r = data.readiness
    if r is None:
        return [
            not_applicable(
                "D02",
                D02_TITLE,
                "Not evaluated: no --required-vram supplied (general inspection).",
                (inference("No --required-vram supplied."),),
            )
        ]
    s = data.snapshot
    evidence: list[Evidence] = [
        observation(
            f"Readiness {r.overall_status} ({r.overall_reason}) for a requirement of "
            f"{r.required_vram_gib:g} GiB ({r.required_bytes} bytes)",
            None,
            "/readiness/overall_status",
        )
    ]
    runtime = {m.device_ref: (i, m) for i, m in enumerate(s.runtime)}
    for j, dev in enumerate(r.devices):
        free_v, total_v = dev.free_bytes.value, dev.total_bytes.value
        free = _gib(free_v) if free_v is not None else f"{dev.free_bytes.status}"
        total = _gib(total_v) if total_v is not None else f"{dev.total_bytes.status}"
        margin = f"{dev.margin_bytes} bytes" if dev.margin_bytes is not None else "n/a"
        mig = ""
        if dev.reason is ReadinessReason.MIG_ENABLED_UNSUPPORTED:
            mig = ", MIG mode enabled"
        source = dev.free_bytes.source
        if source is None and dev.device_ref in runtime:
            source = runtime[dev.device_ref][1].memory_free_bytes.source
        evidence.append(
            Evidence(
                kind=evidence[0].kind,
                statement=(
                    f"{dev.device_ref}: free {free}, total {total}, required "
                    f"{_gib(r.required_bytes)}, margin {margin}{mig} — {dev.status} "
                    f"({dev.reason})"
                ),
                source=source,
                report_path=f"/readiness/devices/{j}",
                value=None,
            )
        )
    limitations = [r.disclaimer, *r.limitations]
    if r.overall_status is ReadinessStatus.READY:
        return [
            finding(
                "D02",
                S.PASS,
                Severity.NONE,
                Confidence.HIGH,
                D02_TITLE,
                f"At least one device had at least {r.required_vram_gib:g} GiB free when sampled.",
                evidence,
                limitations=limitations,
            )
        ]
    if r.overall_status is ReadinessStatus.NOT_READY:
        no_device = r.overall_reason in (
            ReadinessReason.NO_NVIDIA_DEVICE,
            ReadinessReason.PLATFORM_NOT_SUPPORTED,
        )
        all_free = bool(r.devices) and all(d.free_bytes.is_collected for d in r.devices)
        confidence = Confidence.HIGH if all_free and not no_device else Confidence.MEDIUM
        explanation = (
            "No NVIDIA device is available to compare with the requirement."
            if no_device
            else f"No device had {r.required_vram_gib:g} GiB free when sampled."
        )
        return [
            finding(
                "D02",
                S.FAIL,
                Severity.HIGH,
                confidence,
                D02_TITLE,
                explanation,
                evidence,
                recommendations=(
                    advice(
                        "Free GPU memory by closing other GPU workloads (processes observed "
                        "using memory are listed in the runtime section, where visible).",
                        "Free memory is what the comparison uses.",
                    ),
                    advice("Use a GPU with more free memory, if available.", "See above."),
                    advice("Reduce the workload's memory requirement.", "See above."),
                ),
                limitations=limitations,
            )
        ]
    return [
        finding(
            "D02",
            S.UNKNOWN,
            Severity.MEDIUM,
            Confidence.NOT_APPLICABLE,
            D02_TITLE,
            "No NVIDIA device was accessible to measure, and whether one is present couldn't "
            "be determined, so readiness is unknown."
            if r.overall_reason is ReadinessReason.DEVICE_PRESENCE_UNKNOWN
            else "Free memory couldn't be assessed on any device that might qualify, so "
            "readiness is unknown.",
            evidence,
            limitations=limitations,
        )
    ]


# --------------------------------------------------------------------------------- D03

D03_TITLE = "CUDA version landscape"
D03_LIMITATIONS = (
    "Only major-version comparisons are used (CUDA minor version compatibility).",
    "Forward-compatibility (cuda-compat) installations aren't detected.",
    "Toolkit detection relies on version.json and nvcc --version.",
)


def d03(data: DiagnosticInput) -> list[DiagnosticFinding]:
    s = data.snapshot
    drv, env, pt = s.driver, s.cuda_environment, s.frameworks.pytorch
    toolkits = env.toolkits.value or ()
    toolkit_versions = [t.version.value for t in toolkits if t.version.value is not None]
    nvcc_version = env.active_nvcc_version.value
    driver_v = drv.driver_supported_cuda_version.value
    build_v = pt.build_cuda_version.value
    if driver_v is None and not toolkit_versions and nvcc_version is None and build_v is None:
        return [
            not_applicable(
                "D03",
                D03_TITLE,
                "No CUDA version was collected from the driver, a toolkit or PyTorch.",
                (
                    observed_field(
                        "Driver-supported CUDA version",
                        drv.driver_supported_cuda_version,
                        "/driver/driver_supported_cuda_version",
                    ),
                ),
            )
        ]
    toolkit_label = (
        f"CUDA Toolkit {nvcc_version or toolkit_versions[0]} (nvcc)"
        if nvcc_version or toolkit_versions
        else "CUDA Toolkit: none found"
    )
    legend = (
        f"Driver supports CUDA up to {driver_v or 'unknown'} — the newest CUDA version the "
        "installed driver supports, not the installed toolkit. "
        f"{toolkit_label} — a local development toolkit, needed to compile CUDA code, not to "
        "run PyTorch binary packages. "
        f"PyTorch built with CUDA {build_v or 'unknown'} — the CUDA version PyTorch was "
        "built with."
    )
    evidence: list[Evidence] = [
        observed_field(
            "Driver-supported CUDA version",
            drv.driver_supported_cuda_version,
            "/driver/driver_supported_cuda_version",
        ),
        observed_field(
            "Active nvcc version", env.active_nvcc_version, "/cuda_environment/active_nvcc_version"
        ),
        observed_field(
            "PyTorch build CUDA version", pt.build_cuda_version, f"{_PT}/build_cuda_version"
        ),
    ]
    evidence.extend(
        observed_field(
            f"Toolkit {t.root_path} version",
            t.version,
            f"/cuda_environment/toolkits/value/{i}/version",
        )
        for i, t in enumerate(toolkits)
    )
    rows: list[tuple[S, Severity, Confidence, str]] = []
    driver_major = major(driver_v)
    if driver_major is not None and (m := major(build_v)) is not None and m > driver_major:
        rows.append(
            (
                S.WARNING,
                Severity.MEDIUM,
                Confidence.MEDIUM,
                "PyTorch was built with a newer CUDA major version than the driver supports.",
            )
        )
    local = [major(v) for v in (*toolkit_versions, nvcc_version) if v]
    if driver_major is not None and any(m is not None and m > driver_major for m in local):
        rows.append(
            (
                S.WARNING,
                Severity.LOW,
                Confidence.MEDIUM,
                "A local CUDA toolkit is a newer major version than the driver supports.",
            )
        )
    home = [t for t in toolkits if ToolkitDiscovery.CUDA_HOME in t.discovered_via]
    home_mismatch = bool(
        home and nvcc_version and home[0].version.value and home[0].version.value != nvcc_version
    )
    if len(set(toolkit_versions)) > 1 or home_mismatch:
        rows.append(
            (
                S.WARNING,
                Severity.LOW,
                Confidence.HIGH,
                "Several CUDA toolkit versions are present, or CUDA_HOME's toolkit differs "
                "from the active nvcc.",
            )
        )
    if not rows:
        note = (
            " No CUDA toolkit was found; a toolkit is only needed for compiling CUDA code."
            if not toolkit_versions and nvcc_version is None
            else ""
        )
        return [
            finding(
                "D03",
                S.INFO,
                Severity.NONE,
                Confidence.HIGH,
                D03_TITLE,
                legend + note,
                evidence,
                limitations=D03_LIMITATIONS,
            )
        ]
    status, severity, confidence, _ = rows[0]
    evidence.extend(inference(text) for *_, text in rows)
    return [
        finding(
            "D03",
            status,
            severity,
            confidence,
            D03_TITLE,
            legend,
            evidence,
            recommendations=(
                advice(
                    "Keep the driver's supported CUDA major version at or above the toolkit "
                    "and PyTorch build versions, and point CUDA_HOME and PATH at one toolkit.",
                    "Only major-version differences are flagged.",
                    CUDA_COMPAT_URL,
                ),
            ),
            limitations=D03_LIMITATIONS,
        )
    ]


# --------------------------------------------------------------------------------- D04

D04_TITLE = "NVIDIA GPU availability"
D04_STILL_VALID = " The rest of the report is still valid."


def d04(data: DiagnosticInput) -> list[DiagnosticFinding]:
    s = data.snapshot
    drv, gpu = s.driver, s.gpu
    base = [
        observed_field("NVML", drv.nvml_version, "/driver/nvml_version"),
        observed_field("GPU device count", gpu.device_count, "/gpu/device_count"),
        observed_field("nvidia-smi", drv.nvidia_smi_path, "/driver/nvidia_smi_path"),
        observation(
            f"PCI scan is {gpu.pci_observations.status}"
            + (
                f" ({len(ctx.nvidia_pci(s))} NVIDIA device(s))"
                if gpu.pci_observations.is_collected
                else ""
            ),
            gpu.pci_observations,
            "/gpu/pci_observations",
        ),
    ]
    count = ctx.nvml_device_count(s)

    def emit(
        status: S,
        severity: Severity,
        confidence: Confidence,
        explanation: str,
        extra: tuple[Evidence, ...] = (),
        recommendations: tuple[Recommendation, ...] = (),
        limitations: tuple[str, ...] = (),
    ) -> list[DiagnosticFinding]:
        return [
            finding(
                "D04",
                status,
                severity,
                confidence,
                D04_TITLE,
                explanation,
                (*base, *extra),
                recommendations=recommendations,
                limitations=limitations,
            )
        ]

    if count is not None and count >= 1:
        return emit(S.PASS, Severity.NONE, Confidence.HIGH, f"NVML reports {count} device(s).")
    if ctx.fallback_inventory(s) and not drv.nvml_version.is_collected:
        return emit(
            S.PASS,
            Severity.NONE,
            Confidence.MEDIUM,
            f"nvidia-smi reports {len(gpu.devices)} device(s); NVML is unavailable.",
            limitations=("NVML-based fields are unavailable (see D05).",),
        )
    if ctx.is_macos(s):
        return emit(
            S.INFO,
            Severity.NONE,
            Confidence.HIGH,
            "NVIDIA CUDA GPUs are not supported on macOS; general information was "
            "collected." + D04_STILL_VALID,
        )
    nvidia_pci = [(i, o) for i, o in enumerate(gpu.pci_observations.value or ()) if o.is_nvidia]
    if nvidia_pci and not gpu.devices:
        drivers = tuple(
            observed_field(
                f"PCI {o.address} kernel_driver",
                o.kernel_driver,
                f"/gpu/pci_observations/value/{i}/kernel_driver",
            )
            for i, o in nvidia_pci
        )
        return emit(
            S.FAIL,
            Severity.HIGH,
            Confidence.HIGH,
            "NVIDIA GPU present on PCI, but no NVIDIA driver interface is accessible; the "
            "NVML and nvidia-smi sections are unavailable." + D04_STILL_VALID,
            drivers,
            (_driver_advice(s),),
            ("The driver state is inferred from sysfs; other driver problems are possible.",),
        )
    if drv.nvml_version.status is CollectionStatus.PERMISSION_DENIED:
        return emit(
            S.WARNING,
            Severity.MEDIUM,
            Confidence.MEDIUM,
            "NVML was denied access, so NVIDIA sections are unavailable." + D04_STILL_VALID,
            recommendations=(
                advice(
                    "Check the device permissions or the container GPU configuration.",
                    "NVML needs access to the NVIDIA device files.",
                    CONTAINER_TOOLKIT_URL,
                ),
            ),
            limitations=("Permission problems can also come from container isolation.",),
        )
    if count == 0:
        return emit(
            S.WARNING,
            Severity.MEDIUM,
            Confidence.MEDIUM,
            "The NVIDIA driver is loaded but reports no devices." + D04_STILL_VALID,
            recommendations=(_driver_advice(s),),
            limitations=("Devices can be hidden by container or virtualisation settings.",),
        )
    if (
        ctx.pci_scan_complete(s)
        and not nvidia_pci
        and drv.nvml_version.status is CollectionStatus.NOT_DETECTED
        and ctx.smi_absent(s)
    ):
        return emit(
            S.INFO,
            Severity.NONE,
            Confidence.HIGH,
            "No NVIDIA GPU found: the PCI scan found no NVIDIA device and no NVIDIA "
            "driver interface was detected." + D04_STILL_VALID,
        )
    if not ctx.pci_scan_complete(s) and ctx.nvml_absent(s) and ctx.smi_absent(s):
        return emit(
            S.INFO,
            Severity.NONE,
            Confidence.MEDIUM,
            "No NVIDIA GPU is accessible: NVML and nvidia-smi are unavailable and the PCI "
            "scan couldn't be used to confirm hardware presence." + D04_STILL_VALID,
            recommendations=(_driver_advice(s),),
            limitations=(
                "Hardware absence can't be confirmed from this snapshot; a GPU may "
                "exist without an accessible driver interface.",
            ),
        )
    return [
        finding(
            "D04",
            S.UNKNOWN,
            Severity.MEDIUM,
            Confidence.NOT_APPLICABLE,
            D04_TITLE,
            "NVIDIA GPU availability couldn't be determined from the collected sources."
            + D04_STILL_VALID,
            base,
            limitations=("One or more NVIDIA sources failed; see collection errors.",),
        )
    ]


# --------------------------------------------------------------------------------- D05

D05_TITLE = "Dependency availability"
_STATUS_RANK = {S.WARNING: 2, S.INFO: 1, S.PASS: 0}
_SEVERITY_RANK = {Severity.HIGH: 3, Severity.MEDIUM: 2, Severity.LOW: 1, Severity.NONE: 0}


def d05(data: DiagnosticInput) -> list[DiagnosticFinding]:
    s = data.snapshot
    drv, pt = s.driver, s.frameworks.pytorch
    evidence: list[Evidence] = []
    contributions: list[tuple[S, Severity, str]] = []
    recommendations: list[Recommendation] = []
    evidence.append(
        observed_field("nvidia-ml-py (pynvml)", drv.nvml_binding, "/driver/nvml_binding")
    )
    if drv.nvml_binding.status is CollectionStatus.NOT_INSTALLED:
        nvidia_evidence = drv.nvidia_smi_path.is_collected or bool(ctx.nvidia_pci(s))
        explanation = (
            "The required NVML bindings are missing, so the installation is "
            "incomplete. NVML-based fields are unavailable."
        )
        if not nvidia_evidence:
            explanation += " This report is unaffected because no NVIDIA GPU is accessible."
        contributions.append(
            (S.WARNING, Severity.MEDIUM if nvidia_evidence else Severity.LOW, explanation)
        )
        recommendations.append(
            advice(
                "Install nvidia-ml-py into this environment (for example pip install "
                "nvidia-ml-py), or reinstall GPU System Inspector.",
                "nvidia-ml-py is a required dependency.",
                NVML_BINDING_URL,
            )
        )
    dist = drv.nvml_binding_distribution
    if dist.is_collected and dist.value != "nvidia-ml-py":
        evidence.append(
            observed_field("pynvml distribution", dist, "/driver/nvml_binding_distribution")
        )
        contributions.append(
            (
                S.INFO,
                Severity.LOW,
                f"The pynvml module is provided by {dist.value!r}; the documented "
                "recommendation is to depend on nvidia-ml-py.",
            )
        )
    evidence.append(
        observation(
            f"PyTorch section is {pt.status}"
            + (f" ({pt.version.value})" if pt.version.value else ""),
            pt.version if pt.version.is_collected else None,
            f"{_PT}/status",
        )
    )
    if pt.status is CollectionStatus.NOT_INSTALLED:
        contributions.append(
            (
                S.INFO,
                Severity.NONE,
                "PyTorch is not installed in this Python environment; framework "
                "checks were skipped. PyTorch in other environments is not "
                "inspected.",
            )
        )
        recommendations.append(
            advice(
                "Use the official install selector, or run the inspector inside the environment "
                "that has PyTorch. The inspector never installs it.",
                "PyTorch is optional.",
                PYTORCH_SELECTOR_URL,
            )
        )
    elif pt.status is CollectionStatus.SKIPPED:
        contributions.append((S.INFO, Severity.NONE, "PyTorch was skipped by --no-pytorch."))
    evidence.append(observed_field("nvidia-smi", drv.nvidia_smi_path, "/driver/nvidia_smi_path"))
    if (
        drv.nvidia_smi_path.status is CollectionStatus.NOT_DETECTED
        and drv.nvml_version.is_collected
    ):
        contributions.append(
            (
                S.INFO,
                Severity.NONE,
                "nvidia-smi: fallback source unavailable; no impact on this report.",
            )
        )
    if not contributions:
        return [
            finding(
                "D05",
                S.PASS,
                Severity.NONE,
                Confidence.HIGH,
                D05_TITLE,
                "All dependencies the inspector uses are present.",
                evidence,
            )
        ]
    status, severity, _ = max(
        contributions, key=lambda c: (_STATUS_RANK[c[0]], _SEVERITY_RANK[c[1]])
    )
    explanation = " ".join(text for *_, text in contributions)
    limitations: tuple[str, ...] = ()
    if status is S.WARNING:
        limitations = ("Only this Python environment was inspected.",)
    return [
        finding(
            "D05",
            status,
            severity,
            Confidence.HIGH,
            D05_TITLE,
            explanation,
            evidence,
            recommendations=recommendations,
            limitations=limitations,
        )
    ]


# --------------------------------------------------------------------------------- D06

D06_TITLE = "Runtime metrics availability"
_METRIC_FIELDS = (
    "gpu_utilization_percent",
    "memory_bus_utilization_percent",
    "memory_total_bytes",
    "memory_used_bytes",
    "memory_free_bytes",
    "memory_reserved_bytes",
    "temperature_gpu_celsius",
    "temperature_slowdown_threshold_celsius",
    "power_draw_watts",
    "power_limit_watts",
    "processes",
)
_FAILED = (
    CollectionStatus.PERMISSION_DENIED,
    CollectionStatus.COLLECTION_FAILED,
    CollectionStatus.UNKNOWN,
)


def _metric_fields(m: RuntimeMetrics, i: int) -> list[tuple[str, Observed[object], str]]:
    fields: list[tuple[str, Observed[object], str]] = [
        (name, getattr(m, name), f"/runtime/{i}/{name}") for name in _METRIC_FIELDS
    ]
    for j, proc in enumerate(m.processes.value or ()):
        for name in ("used_memory_bytes", "name"):
            fields.append(
                (
                    f"process {proc.pid} {name}",
                    getattr(proc, name),
                    f"/runtime/{i}/processes/value/{j}/{name}",
                )
            )
    return fields


def _platform_reason(s: InspectionSnapshot, m: RuntimeMetrics) -> str:
    if s.system.platform_profile is PlatformProfile.WSL2:
        return " Under WSL2 some NVML queries are not supported."
    if any(
        p.used_memory_bytes.status is CollectionStatus.NOT_SUPPORTED
        for p in m.processes.value or ()
    ):
        return " Per-process memory isn't reported under WDDM."
    return ""


def d06(data: DiagnosticInput) -> list[DiagnosticFinding]:
    s = data.snapshot
    if not s.runtime:
        return [
            not_applicable(
                "D06",
                D06_TITLE,
                "No inventoried devices.",
                (observed_field("GPU device count", s.gpu.device_count, "/gpu/device_count"),),
            )
        ]
    out = []
    for i, m in enumerate(s.runtime):
        fields = _metric_fields(m, i)
        considered = [f for f in fields if f[1].status is not CollectionStatus.SKIPPED]
        lost = [f for f in considered if ctx.error_code(f[1]) == ctx.GPU_IS_LOST]
        failed = [f for f in considered if f[1].status in _FAILED]
        missing = [f for f in considered if not f[1].is_collected]
        skipped = [f for f in fields if f[1].status is CollectionStatus.SKIPPED]
        shown = (lost or failed or missing)[:6]
        evidence = [observed_field(label, o, path) for label, o, path in shown]
        evidence.extend(observed_field(label, o, path) for label, o, path in skipped[:3])
        if not evidence:
            evidence = [
                observation(
                    f"All {len(considered)} runtime fields were collected", None, f"/runtime/{i}"
                )
            ]
        ref = m.device_ref
        if lost:
            out.append(
                finding(
                    "D06",
                    S.FAIL,
                    Severity.HIGH,
                    Confidence.HIGH,
                    D06_TITLE,
                    "NVML reports the GPU as lost; its runtime metrics are unavailable.",
                    evidence,
                    device_ref=ref,
                    recommendations=(
                        advice(
                            "Check the system logs and consider DCGM diagnostics where available.",
                            "A lost GPU needs investigation.",
                            DCGM_URL,
                        ),
                    ),
                    limitations=("The snapshot can't tell why the GPU became unavailable.",),
                )
            )
        elif failed:
            out.append(
                finding(
                    "D06",
                    S.WARNING,
                    Severity.MEDIUM,
                    Confidence.HIGH,
                    D06_TITLE,
                    f"{len(failed)} runtime field(s) couldn't be collected (permission denied or "
                    "query failed).",
                    evidence,
                    device_ref=ref,
                    recommendations=(
                        advice(
                            "Rerun with an account or container configuration that "
                            "has GPU access, if those values are needed.",
                            "The values were denied or failed, not absent.",
                        ),
                    ),
                    limitations=("Other fields of this device are unaffected.",),
                )
            )
        elif missing:
            out.append(
                finding(
                    "D06",
                    S.INFO,
                    Severity.NONE,
                    Confidence.HIGH,
                    D06_TITLE,
                    f"{len(missing)} runtime field(s) are not supported on this device or platform."
                    + _platform_reason(s, m),
                    evidence,
                    device_ref=ref,
                )
            )
        else:
            out.append(
                finding(
                    "D06",
                    S.PASS,
                    Severity.NONE,
                    Confidence.HIGH,
                    D06_TITLE,
                    "All runtime metrics were collected.",
                    evidence,
                    device_ref=ref,
                )
            )
    return out


# --------------------------------------------------------------------------------- D07

D07_TITLE = "Temperature observation"
D07_LIMITATION = (
    "A single temperature sample cannot establish whether the GPU is thermally throttling or "
    "how temperature changes under load (C15). For sustained observation use nvidia-smi "
    "monitoring or DCGM."
)


def d07(data: DiagnosticInput) -> list[DiagnosticFinding]:
    s = data.snapshot
    if not s.runtime:
        return [
            not_applicable(
                "D07",
                D07_TITLE,
                "No inventoried devices.",
                (observed_field("GPU device count", s.gpu.device_count, "/gpu/device_count"),),
            )
        ]
    out = []
    for i, m in enumerate(s.runtime):
        temp, limit = m.temperature_gpu_celsius, m.temperature_slowdown_threshold_celsius
        temp_ev = observed_field(
            "GPU temperature (°C)", temp, f"/runtime/{i}/temperature_gpu_celsius"
        )
        if not temp.is_collected or temp.value is None:
            out.append(
                not_applicable(
                    "D07",
                    D07_TITLE,
                    "Temperature wasn't collected; availability is reported by D06.",
                    (temp_ev,),
                    device_ref=m.device_ref,
                )
            )
            continue
        limit_ev = observed_field(
            "Slowdown threshold (°C)", limit, f"/runtime/{i}/temperature_slowdown_threshold_celsius"
        )
        at = m.sampled_at.isoformat()
        if limit.is_collected and limit.value is not None and temp.value >= limit.value:
            out.append(
                finding(
                    "D07",
                    S.WARNING,
                    Severity.MEDIUM,
                    Confidence.MEDIUM,
                    D07_TITLE,
                    f"Measured {temp.value} °C, at or above the device-reported slowdown threshold "
                    f"of {limit.value} °C at the moment of sampling.",
                    (temp_ev, limit_ev),
                    device_ref=m.device_ref,
                    recommendations=(
                        advice(
                            "Check cooling and observe temperature over time.",
                            "One sample can't show a trend.",
                            DCGM_URL,
                        ),
                    ),
                    limitations=(D07_LIMITATION,),
                )
            )
            continue
        threshold = (
            f"; device-reported slowdown threshold {limit.value} °C" if limit.is_collected else ""
        )
        out.append(
            finding(
                "D07",
                S.INFO,
                Severity.NONE,
                Confidence.HIGH,
                D07_TITLE,
                f"Measured {temp.value} °C at {at}{threshold}.",
                (temp_ev, limit_ev),
                device_ref=m.device_ref,
                limitations=(D07_LIMITATION,),
            )
        )
    return out


# --------------------------------------------------------------------------------- D08

D08_TITLE = "CUDA device visibility and ordering"


def d08(data: DiagnosticInput) -> list[DiagnosticFinding]:
    s = data.snapshot
    devices = s.gpu.devices
    cvd = ctx.env(s, "CUDA_VISIBLE_DEVICES")
    cvd_set = cvd is not None and cvd.is_collected and cvd.value is not None
    cvd_ev = (
        observed_field("CUDA_VISIBLE_DEVICES", cvd, ctx.env_path("CUDA_VISIBLE_DEVICES"))
        if cvd is not None
        else inference("CUDA_VISIBLE_DEVICES was not inspected.")
    )
    context: list[Evidence] = []
    nvd = ctx.env(s, "NVIDIA_VISIBLE_DEVICES")
    if nvd is not None and nvd.is_collected:
        context.append(
            observed_field(
                "NVIDIA_VISIBLE_DEVICES (container context)",
                nvd,
                ctx.env_path("NVIDIA_VISIBLE_DEVICES"),
            )
        )
    env_limits = ("UUID prefixes and MIG IDs are matched best-effort.",)
    if not devices and not cvd_set:
        return [
            not_applicable(
                "D08",
                D08_TITLE,
                "No inventoried devices and CUDA_VISIBLE_DEVICES is unset.",
                (cvd_ev, *context),
            )
        ]
    cvd_advice = advice(
        "Unset or correct CUDA_VISIBLE_DEVICES.",
        "It controls which devices CUDA applications see.",
        CUDA_ENV_VARS_URL,
    )
    if cvd_set and cvd.value is not None and cvd.value.strip() == "":
        return [
            finding(
                "D08",
                S.WARNING,
                Severity.MEDIUM,
                Confidence.HIGH,
                D08_TITLE,
                "CUDA_VISIBLE_DEVICES is empty, so CUDA applications see no devices.",
                (cvd_ev, *context),
                recommendations=(cvd_advice,),
                limitations=env_limits,
            )
        ]
    if not devices:
        return [
            not_applicable(
                "D08",
                D08_TITLE,
                "CUDA_VISIBLE_DEVICES is set, but no inventoried devices exist to compare it with.",
                (cvd_ev, *context),
            )
        ]
    rows: list[tuple[Severity, Confidence, str]] = []
    evidence = [cvd_ev, *context]
    if cvd_set and cvd.value is not None:
        entries = cvd.value.split(",")
        bad = [e.strip() for e in entries if not ctx.visible_entry_matches(e, devices)]
        if bad:
            return [
                finding(
                    "D08",
                    S.WARNING,
                    Severity.MEDIUM,
                    Confidence.MEDIUM,
                    D08_TITLE,
                    f"CUDA_VISIBLE_DEVICES entry {bad[0]!r} matches no observed device; "
                    "CUDA ignores it and every later entry.",
                    evidence,
                    recommendations=(cvd_advice,),
                    limitations=env_limits,
                )
            ]
        rows.append(
            (
                Severity.NONE,
                Confidence.HIGH,
                f"CUDA applications see {len(entries)} device(s), renumbered from 0.",
            )
        )
    mismatched = []
    for i, dev in enumerate(s.frameworks.pytorch.devices):
        ref = dev.correlated_device_ref.value
        if ref and ref.startswith("nvml:") and int(ref.split(":")[1]) != dev.torch_index:
            mismatched.append(f"PyTorch cuda:{dev.torch_index} is NVML device {ref.split(':')[1]}")
            evidence.append(
                observed_field(
                    f"PyTorch cuda:{dev.torch_index} correlation",
                    dev.correlated_device_ref,
                    f"{_PT}/devices/{i}/correlated_device_ref",
                )
            )
    if mismatched:
        rows.append((Severity.LOW, Confidence.HIGH, "; ".join(mismatched) + "."))
    order = ctx.env(s, "CUDA_DEVICE_ORDER")
    names = {d.name.value for d in devices if d.name.value is not None}
    has_correlation = any(
        d.correlated_device_ref.is_collected for d in s.frameworks.pytorch.devices
    )
    if (
        len(devices) >= 2
        and len(names) >= 2
        and not has_correlation
        and (order is None or order.value != "PCI_BUS_ID")
    ):
        if order is not None:
            evidence.append(
                observed_field("CUDA_DEVICE_ORDER", order, ctx.env_path("CUDA_DEVICE_ORDER"))
            )
        rows.append(
            (
                Severity.LOW,
                Confidence.MEDIUM,
                "Device indices may differ between PyTorch and nvidia-smi; set "
                "CUDA_DEVICE_ORDER=PCI_BUS_ID to align them.",
            )
        )
    if not rows:
        return [
            finding(
                "D08",
                S.PASS,
                Severity.NONE,
                Confidence.HIGH,
                D08_TITLE,
                "All inventoried devices are visible to CUDA in their default order.",
                evidence,
            )
        ]
    severity = max((r[0] for r in rows), key=_SEVERITY_RANK.__getitem__)
    return [
        finding(
            "D08", S.INFO, severity, rows[0][1], D08_TITLE, " ".join(r[2] for r in rows), evidence
        )
    ]


DEFAULT_RULES: tuple[RegisteredRule, ...] = (
    RegisteredRule("D01", D01_TITLE, d01),
    RegisteredRule("D02", D02_TITLE, d02),
    RegisteredRule("D03", D03_TITLE, d03),
    RegisteredRule("D04", D04_TITLE, d04),
    RegisteredRule("D05", D05_TITLE, d05),
    RegisteredRule("D06", D06_TITLE, d06),
    RegisteredRule("D07", D07_TITLE, d07),
    RegisteredRule("D08", D08_TITLE, d08),
)
