# 📋 Experimental & Data Governance Protocols (`docs/protocols/`)

Direktori ini memuat protokol formal, pedoman tata kelola data, dan daftar periksa operasional untuk menjamin reproduksibilitas eksperimen rev05.

---

## 📑 Daftar Dokumen Protokol

| Berkas | Ruang Lingkup & Fungsi | Status |
|---|---|:---:|
| [**`EXPERIMENT_PROTOCOL.md`**](EXPERIMENT_PROTOCOL.md) | **Protokol Eksperimen Utama & Rencana Metodologi Selektif:** Pedoman metodologi pengujian decisive validation rev05, skema abstensi adaptif, kalibrasi threshold $\tau$, dan metrik cakupan (*coverage*). | `FROZEN` |
| [**`DATA_PROCESSING_PROTOCOL.md`**](DATA_PROCESSING_PROTOCOL.md) | **Protokol Tata Kelola Data:** Standar pembagian partisi bebas kebocoran (*zero cluster leakage*), pengelompokan `canonical_image_id`, dan penataan pasangan bilingual QA. | `FROZEN` |
| [**`ADAPTIVE_ABSTENTION_IMPLEMENTATION_CHECKLIST.md`**](ADAPTIVE_ABSTENTION_IMPLEMENTATION_CHECKLIST.md) | **Checklist Implementasi Engine:** Verifikasi teknis modul kalibrasi OOF GroupKFold pada Development Set ($K=5$) sebelum eksekusi pada Test Set. | `VERIFIED` |
| [**`EXPERIMENT_UPDATE_CHECKLIST.md`**](EXPERIMENT_UPDATE_CHECKLIST.md) | **Checklist Pra-Eksekusi:** Prosedur verifikasi integritas kode, environment virtual, dan manifest sebelum eksekusi komputasi GPU skala penuh. | `VERIFIED` |

---

## 🔒 Prinsip Integritas Protokol
1. **Pemisahan Klaster Citra Non-Destruktif:** Citra radiologi fisik yang sama dilarang keras muncul di Dev Set dan Test Set secara bersamaan (terverifikasi $0.00\%$ cluster overlap).
2. **Pembekuan Parameter (Frozen Protocol):** Hyperparameter inferensi, token template, dan kandidat sequence scoring dibekukan pada protokol rev05 untuk menjaga validitas komparasi ilmiah.
