# GPU 1 Hardware Fault — Diagnosis & Recovery

**Date:** 2026-03-11
**GPU:** NVIDIA A100 80GB PCIe, Bus ID 0000:46:00.0 (index 1 of 4)
**System:** pc-amax-1

## Symptoms

1. Any CUDA operation on GPU 1 immediately crashes with `torch.AcceleratorError: CUDA error: unknown error`
2. Even `torch._C._cuda_init()` fails — no compute is possible
3. After the crash, GPU 1 becomes invisible to nvidia-smi: `Unable to determine the device handle for GPU1: Unknown Error`
4. **The crash corrupts the entire NVIDIA driver** — other GPUs (0, 2, 3) start failing with Triton/CUDA errors too
5. Only a full system reboot recovers the other GPUs. GPU 1 remains broken after reboot.
6. nvidia-smi shows GPU 1 as healthy (14 MiB used, correct name) immediately after reboot — but it fails the moment any CUDA code touches it

## Timeline

- GPU 1 was working normally before 2026-03-11
- First observed failure: vLLM inference server crash during training rollouts (DP=2, GPU 1 was Engine 001)
- Initially suspected GDN (Gated Delta Net) kernel bug in vLLM + Qwen3.5
- Isolated testing confirmed: GPU 1 can't even initialize CUDA, regardless of workload
- Each test attempt on GPU 1 kills the driver and requires a reboot

## Diagnosis Results (2026-03-11)

### ECC Errors
```
ecc.errors.corrected.volatile.total:      0
ecc.errors.uncorrected.volatile.total:    0
ecc.errors.corrected.aggregate.total:     170
ecc.errors.uncorrected.aggregate.total:   16
```
- **16 uncorrectable aggregate ECC errors** — permanent hardware failures in GPU memory (Double Bit ECC)
- **170 correctable aggregate errors** — Single Bit ECC, less severe but indicate degrading memory
- **Volatile counters are 0** — no new errors since last reboot (the 16 are historical/lifetime)

### Retired Pages
```
retired_pages.address, retired_pages.cause
[N/A], Single Bit ECC
[N/A], Double Bit ECC
```
Pages have been retired for both SBE and DBE causes.

### GPU Reset Attempt
`nvidia-smi --gpu-reset -i 1` fails with "In use by another client" — `nvidia-persistenced` and the `nvidia-modeset` kernel module hold device handles. Stopping the service is not sufficient; `modprobe -r nvidia-modeset` also fails because the module is in use by other kernel modules. **A full reboot is required.**

### Row Remapping Status (post-reboot)
```
Remapped Rows
    Correctable Error                 : 0
    Uncorrectable Error               : 14
    Pending                           : No
    Remapping Failure Occurred        : No
    Bank Remap Availability Histogram
        Max                           : 629 bank(s)
        High                          : 10 bank(s)
        Partial                       : 1 bank(s)
        Low                           : 0 bank(s)
        None                          : 0 bank(s)
```
14 of 16 uncorrectable errors were row-remapped. No remapping failures. 629+ spare banks remain.

### CUDA Test Results (post-reboot)
- **Small tensor (1024x1024)**: PASSES — `torch.randn(1024,1024,device='cuda')` works
- **Stress test (4096x4096 matmul x50)**: CRASHES with `CUDA error: unknown error` during `cuda.synchronize()`
- **Crash impact**: GPU 1 falls off the bus (`Unable to determine the device handle for GPU1`), AND corrupts the entire NVIDIA driver — GPUs 0, 2, 3 all fail with `CUDA unknown error` until the next reboot

### ECC Toggle Results (disable → reboot → re-enable → reboot)
ECC toggle did NOT fix the issue. Same crash on stress test after full disable/re-enable cycle.

### DCGM Level 3 Diagnostic (DCGM 4.5.2, driver 580.105.08)
```
memory              Fail    cuMemGetInfo_v2 failed: 'context is destroyed'
diagnostic          Fail    Xid 79 detected (GPU fallen off bus)
pcie                Fail    All CUDA calls fail (cudaMalloc, cudaEventCreate, etc.)
                            Xid 79 detected (×3)
                            Bandwidth: negative values (GPU unreachable)
memory_bandwidth    Fail    cudaSetDevice failed: device busy/unavailable
targeted_stress     Fail    cudaStreamCreate failed: device busy/unavailable
targeted_power      Fail    cudaStreamCreate failed, Xid 79 detected
```
Every test that touches CUDA fails. The GPU falls off the PCIe bus (Xid 79) under any load.

### What's Been Tried
| Step | Result |
|------|--------|
| Reboot | GPU appears healthy in nvidia-smi but crashes under load |
| Row remapping | 14/16 rows remapped, no failures, 629+ spares — but GPU still crashes |
| ECC toggle (disable → reboot → re-enable → reboot) | No improvement |
| DCGM level 3 diagnostic | ALL tests FAIL, Xid 79 detected |
| nvidia-bug-report.sh | Collected — see analysis below |

### Xid Error History (from nvidia-bug-report.log.gz)
```
Mar 10 21:38:13  Xid 79 (PCI:0000:46:00): GPU has fallen off the bus
Mar 10 23:25:13  Xid 79 (PCI:0000:46:00): pid=14944, name=VLLM::Worker, GPU has fallen off the bus
Mar 11 11:29:29  Xid 79 (PCI:0000:46:00): pid=54555, name=VLLM::Worker, GPU has fallen off the bus
Mar 11 13:00:38  Xid 79 (PCI:0000:46:00): GPU has fallen off the bus
Mar 11 14:29:35  Xid 79 (PCI:0000:46:00): GPU has fallen off the bus  (stress test)
Mar 11 14:55:52  Xid 79 (PCI:0000:46:00): GPU has fallen off the bus  (DCGM diag)
```
Every Xid 79 triggers Xid 154 ("Node Reboot Required") on ALL 4 GPUs — confirming that GPU 1's crash poisons the entire driver.

First crash was **Mar 10 21:38** — during a vLLM training run with Qwen3.5 on DP=2.

### PCIe Link Status
```
LnkCap: Speed 16GT/s, Width x16, ASPM not supported
LnkSta: Speed 16GT/s (ok), Width x16 (ok)
```
PCIe link is healthy — full Gen4 x16, no degradation. No PCIe AER errors in dmesg. This rules out PCIe slot/cable issues as the cause.

### GPU Identification
- **Serial:** 1654023055492
- **VBIOS:** 92.00.A0.00.05
- **Driver:** 580.105.08 (proprietary, not nvidia-open)
- **GSP Firmware:** EnableGpuFirmware = 18 (enabled, default policy)
- **Bug report:** `nvidia-bug-report.log.gz` (saved in this directory)

## Understanding A100 ECC Error Recovery

A100 GPUs use **row remapping** (not just page retirement) to handle ECC errors. The GPU has **640 spare memory rows** that can replace bad ones. Row remapping is triggered at GPU reset or reboot — the driver replaces faulty rows with spare rows in hardware.

Key points:
- **Aggregate errors are lifetime counters** — they accumulate over the GPU's life and cannot be cleared on Ampere+
- **Volatile errors = 0** is a good sign — means no new errors since last reboot
- The 16 uncorrectable errors may have already been successfully remapped in past reboots
- Row remapping happens transparently at reset/reboot; the GPU may still be functional

### RMA Criteria (per NVIDIA docs)
The GPU needs RMA if:
- Row remapping **fails** (Xid 64 in dmesg after reboot)
- A memory bank has exhausted all **8 spare rows** for uncorrectable remaps
- Same errors recur at the same address after remapping
- NVIDIA Field Diagnostic tool (`dcgmi diag`) confirms failure
- More than **4 SRAM uncorrectable errors** within a single address bank

### References
- [NVIDIA A100 GPU Memory Error Management](https://docs.nvidia.com/deploy/a100-gpu-mem-error-mgmt/index.html)
- [Response to Uncorrectable Contained ECC Errors](https://docs.nvidia.com/deploy/a100-gpu-mem-error-mgmt/response-to-uncorrectable-contained-ecc-errors.html)
- [Dynamic Page Retirement](https://docs.nvidia.com/deploy/dynamic-page-retirement/index.html)
- [stas00/ml-engineering GPU Debug Guide](https://github.com/stas00/ml-engineering/blob/master/compute/accelerator/nvidia/debug.md)

## Remaining Recovery Steps To Try

Steps 1–6 have been completed (reboot, row remapping check, ECC toggle, DCGM diagnostic). The following steps have NOT been tried yet.

### 7. Disable GSP Firmware
GSP (GPU System Processor) firmware runs on the GPU and handles driver communication. Disabling it forces the CPU-side driver to handle everything directly. This has fixed Xid 79 and GPU crash issues for some users.

**Only works with the proprietary NVIDIA driver** (not nvidia-open).

```bash
# Check if using proprietary driver
cat /proc/driver/nvidia/version

# Check current GSP status
cat /proc/driver/nvidia/params | grep EnableGpuFirmware

# Disable GSP
echo "options nvidia NVreg_EnableGpuFirmware=0" | sudo tee /etc/modprobe.d/nvidia-gsp.conf
sudo reboot

# Verify after reboot
cat /proc/driver/nvidia/params | grep EnableGpuFirmware
# Should show: EnableGpuFirmware: 0

# Test GPU 1
CUDA_VISIBLE_DEVICES=1 python3 -c "import torch; a=torch.randn(4096,4096,device='cuda'); b=torch.randn(4096,4096,device='cuda'); [torch.mm(a,b) for _ in range(50)]; torch.cuda.synchronize(); print('OK')"

# IMPORTANT: If this doesn't help, remove the config to re-enable GSP
sudo rm /etc/modprobe.d/nvidia-gsp.conf
sudo reboot
```

References:
- [NVIDIA GSP Firmware Documentation](https://download.nvidia.com/XFree86/Linux-x86_64/510.39.01/README/gsp.html)
- [Arch Linux — Solved: Disabling GSP firmware](https://bbs.archlinux.org/viewtopic.php?id=300747)
- [AWS re:Post — Troubleshoot Xid errors](https://repost.aws/knowledge-center/ec2-linux-troubleshoot-xid-errors)

### 8. Disable PCIe ASPM (Active State Power Management)
PCIe power management can cause link instability, leading to Xid 79. Disabling ASPM has resolved "GPU fallen off the bus" for multiple users.

```bash
# Option A: Kernel parameter (persistent across reboots)
# Add to GRUB_CMDLINE_LINUX in /etc/default/grub:
#   pcie_aspm=off
sudo sed -i 's/GRUB_CMDLINE_LINUX="\(.*\)"/GRUB_CMDLINE_LINUX="\1 pcie_aspm=off"/' /etc/default/grub
sudo update-grub
sudo reboot

# Option B: Runtime (immediate, not persistent)
sudo setpci -s 46:00.0 CAP_EXP+10.w=0040

# Verify ASPM is off
sudo lspci -vvs 46:00.0 | grep -i aspm
```

Can be combined with GSP disable: `nvidia.NVreg_EnableGpuFirmware=0 pcie_aspm=off` in kernel cmdline.

References:
- [Arch Linux — Solved: GPU fallen off the bus](https://bbs.archlinux.org/viewtopic.php?id=304020)
- [NVIDIA Forums — Fix Xid 79](https://forums.developer.nvidia.com/t/fix-xid-79-gpu-has-fallen-off-the-bus-already/165574)

### 9. Check PCIe Link Health
Xid 79 can be caused by PCIe bus errors rather than GPU memory issues. Check if the PCIe link itself is degraded.

```bash
# Check PCIe link speed and width (should be Gen4 x16 for A100 PCIe)
sudo lspci -vvs 46:00.0 | grep -E "LnkSta|LnkCap|Width|Speed"

# Check for PCIe AER (Advanced Error Reporting) errors
sudo dmesg | grep -i "aer\|pci.*error\|46:00"

# Check PCIe power
sudo lspci -vvs 46:00.0 | grep -i power
```

If link is running at lower speed/width than expected (e.g., Gen3 instead of Gen4, or x8 instead of x16), this points to a PCIe slot/cable issue rather than GPU memory.

### 10. Collect NVIDIA Bug Report (for RMA filing)
```bash
sudo nvidia-bug-report.sh
# Creates nvidia-bug-report.log.gz in current directory
```
This captures driver state, Xid errors, PCIe topology, ECC status — everything needed for an RMA ticket. AMAX/NVIDIA will likely ask for this.

You can also use Lambda's parser for a quick summary:
```bash
git clone https://github.com/lambdal-support/lambda-public-tools.git
bash lambda-public-tools/check-nvidia-bug-report.sh nvidia-bug-report.log
```

### 11. Physical Inspection
- **Reseat GPU 1** in its PCIe slot, clean gold contacts with isopropyl alcohol
- **Check power cables** — A100 PCIe needs 2x 8-pin power connectors on independent 12V rails
- **Swap power cables** with a known-good GPU's cables
- **Try GPU 1 in a different PCIe slot**:
  - If fault follows the GPU → GPU hardware defect, needs RMA
  - If fault stays with the slot → motherboard/riser/cable issue

### 12. RMA
Contact AMAX for warranty replacement. Provide:
- ECC error counts (16 uncorrectable, 170 correctable aggregate)
- DCGM `dcgmi diag -r 3` results (all FAIL, Xid 79)
- `nvidia-bug-report.log.gz`
- Xid errors from dmesg
- GPU serial number: `nvidia-smi -i 1 --query-gpu=serial --format=csv,noheader`

References:
- [NVIDIA GPU Debug Guidelines](https://docs.nvidia.com/deploy/gpu-debug-guidelines/index.html)
- [NVIDIA Xid Errors Documentation](https://docs.nvidia.com/deploy/xid-errors/analyzing-xid-catalog.html)
- [Lambda — Using nvidia-bug-report.log](https://docs.lambda.ai/education/linux-usage/using-the-nvidia-bug-report.log-file-to-troubleshoot-your-system/)
- [Exxact — GPU Troubleshooting: Resolving ECC Errors](https://support.exxactcorp.com/hc/en-us/articles/30923566578071-GPU-Troubleshooting-Guide-Resolving-ECC-Errors)

## Current Workaround

Training runs on GPUs 0, 2, 3 only:
```bash
CUDA_VISIBLE_DEVICES=0,2,3 uv run rl @ configs/arc_agi/opd-rl-qwen3.5-9b.toml
```
Config uses `num_infer_gpus = 1` (single GPU inference instead of DP=2).

**CRITICAL: Do not run any CUDA operations on GPU 1.** Even a single `torch.cuda.init()` on GPU 1 corrupts the entire NVIDIA driver, crashing all 4 GPUs and requiring a full reboot.
