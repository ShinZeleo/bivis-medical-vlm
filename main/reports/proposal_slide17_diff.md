# Dokumen Perubahan (Diff): Slide 17 Proposal Penelitian Pasca-Persetujuan Dosen

**Berkas Sumber:** `Medical_VLM_Proposal.pptx` (Asli, dipertahankan)  
**Berkas Revisi:** `Medical_VLM_Proposal_Revisi_Dosen.pptx` (Salinan revisi dosen)  
**Target:** Slide 17 ("Sinyal Visual & Intervensi")  
**Tanggal Revisi:** 18 Oktober 2026  

---

## 1. Ringkasan Perubahan Berdasarkan Keputusan Dosen

Sesuai arahan dosen pembimbing bahwa **blur ditiadakan (karena kualitas citra sudah baik dan tidak diperlukan)** serta **intervensi citra blank hitam disetujui**, Slide 17 telah diselaraskan dengan implementasi eksperimen yang sebenarnya.

---

## 2. Tabel Perbandingan Elemen Slide 17 (Diff)

| Elemen Bentuk (*Shape*) | Teks Asli (`Medical_VLM_Proposal.pptx`) | Teks Revisi (`Medical_VLM_Proposal_Revisi_Dosen.pptx`) | Rationale / Alasan Perubahan |
| :--- | :--- | :--- | :--- |
| **Judul Slide (`Text 0`)** | `Sinyal Visual: Tiga Ukuran yang Dibedakan` | `Sinyal Visual: Dua Kondisi Utama & Evaluasi Berarah` | Menyesuaikan fokus ke dua kondisi utama ($P_0$ dan $P_1$) sesuai arahan dosen. |
| **Kondisi 1 (`Text 2`)** | `P0: gambar asli` | `P0: gambar asli (radiologi)` | Memperjelas modalitas radiologi asli. |
| **Kondisi 2 (`Text 4`)** | `P1: gambar blank` | `P1: gambar blank hitam (disetujui dosen)` | Menegaskan kanvas hitam deterministik yang telah disetujui dosen. |
| **Kondisi 3 (`Text 6`)** | `P2: pixel-shuffled` | `(P2: pixel-shuffled, opsional)` | Memindahkan pixel-shuffled menjadi opsional sekunder, menghapus dari alur primer. |
| **Penjelasan VFR (`Text 9`)** | `Proporsi jawaban yang berubah saat gambar diintervensi.` | `Proporsi jawaban yang berubah saat gambar dihitamkan.` | Spesifik pada penghapusan modalitas visual via citra hitam. |
| **Metrik Sensitivitas (`Text 11`)**| `Perubahan confidence` | `Sensitivitas visual berarah` | Mengganti konsep selisih absolut dengan sensitivitas berarah (*directed*). |
| **Rumus Sensitivitas (`Text 12`)**| `\|confidence asli − confidence intervensi\|` | `p_orig(y_hat) - p_blank(y_hat)` | Mengunci rumus $\Delta_{\text{dir}}$ berarah (menolak formula unsigned yang cacat). |
| **Metrik Seleksi (`Text 14`)** | `Risk–coverage` | `Risk-coverage & Discrete AURC` | Menyebutkan metrik integrasi diskrit standar secara eksplisit. |
| **Deskripsi Seleksi (`Text 15`)**| `Risiko kesalahan terhadap cakupan jawaban.` | `Risiko kesalahan terhadap cakupan jawaban selektif.` | Terminologi baku *selective classification*. |
| **Catatan Kritis (`Text 16`)** | `Proxy perilaku, bukan bukti clinical grounding. Gambar blank hitam dapat menjadi input di luar distribusi; metode intervensi perlu divalidasi dan metrik final dikunci setelah pilot.` | `Proxy perilaku, bukan bukti clinical grounding. Sesuai arahan dosen, blur ditiadakan (kualitas citra sudah baik). Gambar blank hitam menjadi intervensi utama; evaluasi membuktikan kemampuan perankingan risiko kesalahan, bukan jaminan keselamatan klinis.` | Mencantumkan keputusan dosen mengenai peniadaan blur dan mempertegas batasan keselamatan klinis. |

---

## 3. Verifikasi Keberadaan Blur di Seluruh Slide
Pencarian teks menyeluruh (*full-text search*) pada seluruh slide XML `Medical_VLM_Proposal.pptx` mengonfirmasi bahwa **kata "blur" tidak muncul di slide manapun**. Dengan demikian, tidak ada sisa penyebutan blur yang tertinggal pada proposal.
