# References

Standards, papers and books behind the physical models of LinkStudio. The model-by-model review
is in [docs/PHYSICS_REVIEW.md](../docs/PHYSICS_REVIEW.md); keys in square brackets match it.

**Licensing.** The PDFs in this folder are free to download from their publishers (ITU, ETSI,
arXiv, the authors' pages, the Bell System Technical Journal archive). They are **not**
redistributed: `.gitignore` excludes `references/**/*.pdf`, so every clone has to fetch them
again with `python references/fetch.py`, which also cuts the classic Bell System Technical
Journal articles out of the full issues.

Status column:

- **in folder**: downloaded into the given subfolder.
- **download**: paywalled or behind a login. Fetch it yourself (IEEE Xplore, an institutional
  login, or the IEEE GET program) and save it under the suggested name.

## Wireless (`wireless/`)

| Key | Reference | Status |
|-----|-----------|--------|
| [36.104] | 3GPP TS 36.104 v19.2.0, *E-UTRA BS radio transmission and reception*, Annex B (EPA/EVA/ETU). [ETSI](https://www.etsi.org/deliver/etsi_ts/136100_136199/136104/19.02.00_60/ts_136104v190200p.pdf) | in folder |
| [38.901] | 3GPP TR 38.901 v19.1.0, *Channel model for frequencies from 0.5 to 100 GHz* (path loss, TDL/CDL). [ETSI](https://www.etsi.org/deliver/etsi_tr/138900_138999/138901/19.01.00_60/tr_138901v190100p.pdf) | in folder |
| [M.1225] | ITU-R M.1225, *Guidelines for evaluation of radio transmission technologies for IMT-2000* (1997). [ITU](https://www.itu.int/rec/R-REC-M.1225/en) | in folder |
| [P.525] | ITU-R P.525-5, *Calculation of free-space attenuation* (2024). [ITU](https://www.itu.int/rec/R-REC-P.525/en) | in folder |
| [Clarke 1968] | R. H. Clarke, "A statistical theory of mobile-radio reception," *Bell Syst. Tech. J.* 47(6), 957–1000, 1968 (cut from the full issue). | in folder |
| [Tse] | D. Tse, P. Viswanath, *Fundamentals of Wireless Communication*, Cambridge UP, 2005: chapters 2, 3 and 7 from the [authors' page](https://web.stanford.edu/~dntse/wireless_book.html) | in folder |
| [Proakis] | J. G. Proakis, M. Salehi, *Digital Communications*, 5th ed., McGraw-Hill, 2008 (book) | download |
| [Zheng & Xiao 2003] | Y. R. Zheng, C. Xiao, "Simulation models with correct statistical properties for Rayleigh fading channels," *IEEE Trans. Commun.* 51(6), 2003. [doi:10.1109/TCOMM.2003.813259](https://doi.org/10.1109/TCOMM.2003.813259) | download |
| [Saleh 1981] | A. A. M. Saleh, "Frequency-independent and frequency-dependent nonlinear models of TWT amplifiers," *IEEE Trans. Commun.* 29(11), 1981. [doi:10.1109/TCOM.1981.1094911](https://doi.org/10.1109/TCOM.1981.1094911) | download |
| [Tarighat et al. 2005] | A. Tarighat, R. Bagheri, A. H. Sayed, "Compensation schemes and performance analysis of IQ imbalances in OFDM receivers," *IEEE Trans. Signal Process.* 53(8), 2005. [doi:10.1109/TSP.2005.851156](https://doi.org/10.1109/TSP.2005.851156) | download |

## Copper (`wireline/`)

| Key | Reference | Status |
|-----|-----------|--------|
| [G.996.1] | ITU-T G.996.1, *Test procedures for DSL transceivers* (test loops, cable models), 2001. [ITU](https://www.itu.int/rec/T-REC-G.996.1/en) | in folder |
| [G.992.1] | ITU-T G.992.1, *ADSL transceivers*, 1999. [ITU](https://www.itu.int/rec/T-REC-G.992.1/en) | in folder |
| [G.993.2] | ITU-T G.993.2, *VDSL2 transceivers*, 2019. [ITU](https://www.itu.int/rec/T-REC-G.993.2/en) | in folder |
| [Cioffi ch. 3, 4] | J. M. Cioffi, *Data Transmission Theory*, ch. 3 (equalization) and ch. 4 (multichannel modulation, gap, loading). [Course reader](https://cioffi-group.stanford.edu/ee379b/course_reader.html) | in folder |
| [ANSI T1.417] | ANSI T1.417-2003, *Spectrum Management for Loop Transmission Systems* (NEXT/FEXT models). Sold by ATIS. | download |
| [Starr–Cioffi–Silverman] | T. Starr, J. M. Cioffi, P. Silverman, *Understanding Digital Subscriber Line Technology*, Prentice Hall, 1999 (book) | download |
| [Djordjevic et al. 2001] | A. R. Djordjevic et al., "Wideband frequency-domain characterization of FR-4 and time-domain causality," *IEEE Trans. EMC* 43(4), 2001. [doi:10.1109/15.974647](https://doi.org/10.1109/15.974647) | download |
| [IEEE 802.3] | IEEE Std 802.3-2022: clause 24 (4B/5B), 49 (64B/66B), Annex 69B (backplane channel), 52/95/138 (multimode PMDs). Free through the IEEE GET program on IEEE Xplore (Browse, Standards, GET Program) after an IEEE login. | download |
| [TIA-568] | ANSI/TIA-568.2-D (balanced cabling) and TIA-568.3-D (optical fibre cabling) | download |

## Optical fibre and amplifiers (`optical/`)

| Key | Reference | Status |
|-----|-----------|--------|
| [G.652] | ITU-T G.652, *Characteristics of a single-mode optical fibre and cable*, 2024. [ITU](https://www.itu.int/rec/T-REC-G.652/en) | in folder |
| [G.694.1] | ITU-T G.694.1, *DWDM frequency grid*, 2020. [ITU](https://www.itu.int/rec/T-REC-G.694.1/en) | in folder |
| [G.662] | ITU-T G.662, *Generic characteristics of optical amplifier devices and subsystems*, 2005. [ITU](https://www.itu.int/rec/T-REC-G.662/en) | in folder |
| [G.Sup39] | ITU-T G-series Supplement 39, *Optical system design and engineering considerations*, 2025. [ITU](https://www.itu.int/rec/T-REC-G.Sup39/en) | in folder |
| [Personick 1973] | S. D. Personick, "Receiver design for digital fiber optic communication systems, I and II," *Bell Syst. Tech. J.* 52(6), 843–886, 1973 (cut from the full issue). | in folder |
| [Agrawal NLFO] | G. P. Agrawal, *Nonlinear Fiber Optics*, 6th ed., Academic Press, 2019 (book) | download |
| [Agrawal FOCS] | G. P. Agrawal, *Fiber-Optic Communication Systems*, 5th ed., Wiley, 2021 (book) | download |
| [Sinkin 2003] | O. V. Sinkin et al., "Optimization of the split-step Fourier method in modeling optical-fiber communications systems," *J. Lightw. Technol.* 21(1), 2003. [doi:10.1109/JLT.2003.808628](https://doi.org/10.1109/JLT.2003.808628) | download |
| [Marcuse et al. 1997] | D. Marcuse, C. R. Menyuk, P. K. A. Wai, "Application of the Manakov-PMD equation to studies of signal propagation in optical fibers with randomly varying birefringence," *J. Lightw. Technol.* 15(9), 1997. [doi:10.1109/50.622902](https://doi.org/10.1109/50.622902) | download |
| [Poggiolini 2014] | P. Poggiolini et al., "The GN-model of fiber non-linear propagation and its applications," *J. Lightw. Technol.* 32(4), 2014. [doi:10.1109/JLT.2013.2295208](https://doi.org/10.1109/JLT.2013.2295208) | download |
| [Kikuchi 2016] | K. Kikuchi, "Fundamentals of coherent optical fiber communications," *J. Lightw. Technol.* 34(1), 2016. [doi:10.1109/JLT.2015.2463719](https://doi.org/10.1109/JLT.2015.2463719) | download |
| [Savory 2010] | S. J. Savory, "Digital coherent optical receivers: algorithms and subsystems," *IEEE JSTQE* 16(5), 2010. [doi:10.1109/JSTQE.2010.2044751](https://doi.org/10.1109/JSTQE.2010.2044751) | download |
| [McIntyre 1966] | R. J. McIntyre, "Multiplication noise in uniform avalanche diodes," *IEEE Trans. Electron Devices* 13(1), 1966. [doi:10.1109/T-ED.1966.15651](https://doi.org/10.1109/T-ED.1966.15651) | download |
| [Koch & Bowers 1984] | T. L. Koch, J. E. Bowers, "Nature of wavelength chirping in directly modulated semiconductor lasers," *Electron. Lett.* 20(25), 1984 | download |

## Short reach: multimode fibre and VCSELs (`optical/multimode/`)

| Key | Reference | Status |
|-----|-----------|--------|
| [G.651.1] | ITU-T G.651.1, *Characteristics of a 50/125 µm multimode graded index optical fibre cable*, 2018. [ITU](https://www.itu.int/rec/T-REC-G.651.1/en) | in folder |
| [Cunningham & Dawe 2002] | D. Cunningham, P. Dawe, *Review of the 10 Gigabit Ethernet link model* (white paper, Agilent/Avago). The IEEE 802.3 link model: Gaussian modal and chromatic bandwidths, ISI and noise penalties. [Broadcom](https://docs.broadcom.com/doc/AV02-2485EN) | in folder |
| [Gloge & Marcatili 1973] | D. Gloge, E. A. J. Marcatili, "Multimode theory of graded-core fibers," *Bell Syst. Tech. J.* 52(9), 1563–1578, 1973 (cut from the full issue). | in folder |
| [Olshansky & Keck 1976] | R. Olshansky, D. B. Keck, "Pulse broadening in graded-index optical fibers," *Appl. Opt.* 15(2), 483–491, 1976. [doi:10.1364/AO.15.000483](https://doi.org/10.1364/AO.15.000483) | download |
| [Golowich et al. 2003] | S. E. Golowich, P. Pepeljugoski et al., "Modeling and simulation of next-generation multimode fiber links," *J. Lightw. Technol.* 21(5), 2003. [doi:10.1109/JLT.2003.811320](https://doi.org/10.1109/JLT.2003.811320) | download |
| [Pepeljugoski et al. 2003] | P. Pepeljugoski et al., "Development of system specification for laser-optimized 50-µm multimode fiber for multigigabit short-wavelength LANs," *J. Lightw. Technol.* 21(5), 2003. [doi:10.1109/JLT.2003.811321](https://doi.org/10.1109/JLT.2003.811321) | download |
| [Larsson 2011] | A. Larsson, "High-speed VCSELs for short reach communication," *Semicond. Sci. Technol.* 26(1), 014017, 2011. [doi:10.1088/0268-1242/26/1/014017](https://doi.org/10.1088/0268-1242/26/1/014017) | download |
| [Coldren] | L. A. Coldren, S. W. Corzine, M. L. Mašanović, *Diode Lasers and Photonic Integrated Circuits*, 2nd ed., Wiley, 2012, ch. 5 (rate equations, small-signal response) (book) | download |
| [IEC 60793-2-10] | IEC 60793-2-10, *Sectional specification for category A1 multimode fibres* (OM classes, DMD/EMB) | download |
| [ISO/IEC 11801] | ISO/IEC 11801-1, *Generic cabling for customer premises* (OM1–OM5 bandwidths) | download |

## Free-space optics (`optical/fso/`)

| Key | Reference | Status |
|-----|-----------|--------|
| [P.1814] | ITU-R P.1814-1, *Prediction methods required for the design of terrestrial free-space optical links*, 2025. [ITU](https://www.itu.int/rec/R-REC-P.1814/en) | in folder |
| [P.1817] | ITU-R P.1817-1, *Propagation data required for the design of terrestrial free-space optical links*, 2012. [ITU](https://www.itu.int/rec/R-REC-P.1817/en) | in folder |
| [Kaushal & Kaddoum 2015] | H. Kaushal, G. Kaddoum, "Free space optical communication: challenges and mitigation techniques," 2015. [arXiv:1506.04836](https://arxiv.org/abs/1506.04836) | in folder |
| [Al-Habash et al. 2001] | M. A. Al-Habash, L. C. Andrews, R. L. Phillips, "Mathematical model for the irradiance probability density function of a laser beam propagating through turbulent media," *Opt. Eng.* 40(8), 2001. [doi:10.1117/1.1386641](https://doi.org/10.1117/1.1386641) | download |
| [Farid & Hranilovic 2007] | A. A. Farid, S. Hranilovic, "Outage capacity optimization for free-space optical links with pointing errors," *J. Lightw. Technol.* 25(7), 2007. [doi:10.1109/JLT.2007.899174](https://doi.org/10.1109/JLT.2007.899174) | download |
| [Khalighi & Uysal 2014] | M. A. Khalighi, M. Uysal, "Survey on free space optical communication: a communication theory perspective," *IEEE Commun. Surveys Tuts.* 16(4), 2014. [doi:10.1109/COMST.2014.2329501](https://doi.org/10.1109/COMST.2014.2329501) | download |
