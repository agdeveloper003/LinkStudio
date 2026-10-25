"""Download the freely available references into references/<topic>/.

Usage: python references/fetch.py   (needs pypdf to cut articles out of journal issues)
"""

import io
import urllib.request
from pathlib import Path

import pypdf

ROOT = Path(__file__).resolve().parent
UA = {"User-Agent": "Mozilla/5.0"}


def itu(rec_id: str) -> str:
    if rec_id.startswith("R-REC-"):
        series = rec_id.split("-")[2].split(".")[0].lower()
        return f"https://www.itu.int/dms_pubrec/itu-r/rec/{series}/{rec_id}!!PDF-E.pdf"
    return f"https://www.itu.int/rec/dologin_pub.asp?lang=e&id={rec_id}!!PDF-E&type=items"


BSTJ = "https://www.worldradiohistory.com/Archive-Bell-System-Technical-Journal/"
FILES = [
    # (folder, file name, url)
    ("optical/multimode", "ITU-T_G.651.1_2018_multimode_fibre_50um.pdf", itu("T-REC-G.651.1-201811-I")),
    ("optical/multimode", "Cunningham_Dawe_2002_10GbE_link_model_review.pdf", "https://docs.broadcom.com/doc/AV02-2485EN"),
    ("optical", "ITU-T_G.652_2024_single_mode_fibre.pdf", itu("T-REC-G.652-202408-I")),
    ("optical", "ITU-T_G.694.1_2020_DWDM_frequency_grid.pdf", itu("T-REC-G.694.1-202010-I")),
    ("optical", "ITU-T_G.662_2005_optical_amplifiers.pdf", itu("T-REC-G.662-200507-I")),
    ("optical", "ITU-T_G.Sup39_2025_optical_system_design.pdf", itu("T-REC-G.Sup39-202510-I")),
    ("optical/fso", "ITU-R_P.1814-1_2025_FSO_attenuation_prediction.pdf", itu("R-REC-P.1814-1-202509-I")),
    ("optical/fso", "ITU-R_P.1817-1_2012_FSO_propagation_data.pdf", itu("R-REC-P.1817-1-201202-I")),
    ("optical/fso", "Kaushal_Kaddoum_2015_FSO_challenges_mitigation_arXiv1506.04836.pdf", "https://arxiv.org/pdf/1506.04836"),
    ("wireless", "3GPP_TS36.104_v19.2.0_LTE_BS_radio_AnnexB_channel_models.pdf",
     "https://www.etsi.org/deliver/etsi_ts/136100_136199/136104/19.02.00_60/ts_136104v190200p.pdf"),
    ("wireless", "3GPP_TR38.901_v19.1.0_channel_model_0.5-100GHz.pdf",
     "https://www.etsi.org/deliver/etsi_tr/138900_138999/138901/19.01.00_60/tr_138901v190100p.pdf"),
    ("wireless", "ITU-R_M.1225_1997_IMT-2000_evaluation_channel_models.pdf", itu("R-REC-M.1225-0-199702-I")),
    ("wireless", "ITU-R_P.525-5_2024_free_space_attenuation.pdf", itu("R-REC-P.525-5-202411-I")),
    ("wireless", "Tse_Viswanath_ch2_the_wireless_channel.pdf",
     "https://web.stanford.edu/~dntse/Chapters_PDF/Fundamentals_Wireless_Communication_chapter2.pdf"),
    ("wireless", "Tse_Viswanath_ch3_detection_diversity.pdf",
     "https://web.stanford.edu/~dntse/Chapters_PDF/Fundamentals_Wireless_Communication_chapter3.pdf"),
    ("wireless", "Tse_Viswanath_ch7_MIMO_spatial_multiplexing.pdf",
     "https://web.stanford.edu/~dntse/Chapters_PDF/Fundamentals_Wireless_Communication_chapter7.pdf"),
    ("wireline", "ITU-T_G.996.1_2001_DSL_test_procedures_loops.pdf", itu("T-REC-G.996.1-200102-I")),
    ("wireline", "ITU-T_G.992.1_1999_ADSL.pdf", itu("T-REC-G.992.1-199907-I")),
    ("wireline", "ITU-T_G.993.2_2019_VDSL2.pdf", itu("T-REC-G.993.2-201902-I")),
    ("wireline", "Cioffi_ch4_multichannel_modulation.pdf", "https://cioffi-group.stanford.edu/doc/book/chap4.pdf"),
    ("wireline", "Cioffi_ch3_equalization.pdf", "https://cioffi-group.stanford.edu/doc/book/chap3.pdf"),
]

# Articles cut out of full Bell System Technical Journal issues: (folder, name, issue url, title words, pages)
ARTICLES = [
    ("wireless", "Clarke_1968_BSTJ_statistical_theory_mobile_radio_reception.pdf",
     BSTJ + "60s/Bell-System-Technical-Journal-1968-6-Complete.pdf", "statistical theory of mobile", 44),
    ("optical/multimode", "Gloge_Marcatili_1973_BSTJ_multimode_theory_graded_core_fibers.pdf",
     BSTJ + "70s/Bell-System-Technical-Journal-1973-9.pdf", "multimode theory of graded", 16),
    ("optical", "Personick_1973_BSTJ_receiver_design_fiber_optic_systems.pdf",
     BSTJ + "70s/Bell-System-Technical-Journal-1973-6.pdf", "receiver design for digital fiber", 44),
]


def get(url: str) -> bytes:
    if "arxiv.org" in url:  # the Python certificate store lacks arXiv's issuer here
        import subprocess
        return subprocess.run(["curl", "-sL", "-A", "Mozilla/5.0", url], capture_output=True, check=True).stdout
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=180) as r:
        return r.read()


def save(folder: str, name: str, data: bytes) -> None:
    if not data.startswith(b"%PDF"):
        raise ValueError("not a PDF")
    path = ROOT / folder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    print(f"ok   {folder}/{name} ({len(data) / 1e6:.1f} MB)")


for folder, name, url in FILES:
    if (ROOT / folder / name).exists():
        continue
    try:
        save(folder, name, get(url))
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL {folder}/{name}: {exc}")

for folder, name, url, words, n_pages in ARTICLES:
    if (ROOT / folder / name).exists():
        continue
    try:
        reader = pypdf.PdfReader(io.BytesIO(get(url)))
        start = next(i for i, page in enumerate(reader.pages)
                     if words.replace(" ", "") in "".join((page.extract_text() or "").lower().split()))
        writer = pypdf.PdfWriter()
        for i in range(start, min(start + n_pages, len(reader.pages))):
            writer.add_page(reader.pages[i])
        buf = io.BytesIO()
        writer.write(buf)
        save(folder, name, buf.getvalue())
        print(f"     (issue pages {start + 1}-{start + n_pages})")
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL {folder}/{name}: {exc}")
