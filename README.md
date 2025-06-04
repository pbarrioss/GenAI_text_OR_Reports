**OR Report Geenerator**
Gen AI project - generates full OR report form simple prompt

I wanted to build a tool that generates surgical reports for simple prompts. Takes a brief description and outputs a detailed operative note. For the sake of focused simplicity, I used only appendectomy for now. 

My approach fine-tunes BioGPT on the operative report data to generate full surgical reports from brief clinical descriptions. It includes a web interface with adjustable settings via GUI.

How to use:
Prompts like
- laparasocopic uncomplicated appendicitis
- open perforated appendicitis 

I initially used GPT-2 and it was making some very strange answers - quite frankly they were comical. Then I switch to BioGPT, trained on actual medical papers (15M+ PubMed documents), knows surgical terminology natively, understands medical procedure flow, and generates proper operative note structure. I am happy I found this. 

EXAMPLE:
============================================================
TEST 4: Laparoscopic appendectomy without perforation in young patient
============================================================
The patient was prepped and draped in sterile fashion. A Foley catheter was inserted for bladder decompression prior to incision of the abdomen. After obtaining an adequate general endotracheal anesthetic, a 5-mm port placed at the level of umbilicus under direct vision. A Veress needle was introduced into the right lower quadrant. An EndoGIA stapler was used to create a 10-mm umbilical defect. Pneumoperitoneum was performed with CO2 insufflation up to 15 mmHg. Once hemostasis had been obtained, the fascia was incised along its entire length. Next, a 12-mm trocar was inserted suprapubic. Then, two additional 5-0 Vicryl subcuticular sutures were applied across the midline. Finally, the appendix stump was ligated using the Endo GIA device. All ports were removed under direct visualization. The peritoneum was irrigated with warm saline solution. A 4 x 2 cm area of the anterior abdominal wall was explored through which no bleeding or leakage from the wound occurred. The skin incisions were closed with 3-0 Monocryl subcostal stitch. The fascia was approximated with interrupted 0 Vicia monofilament suture. The remaining layers were approximated with Dermabond. Steri-Strips. The peritoneal cavity was irrigated and then all drains were removed. The patient tolerated this well. She was extubated immediately after surgery and returned to recovery room in stable condition. On postoperative day 1, she was taken back to recovery ward where there were no signs of infection. On follow-up examination on postoperative day 7, her physical exam revealed normal vital signs, and she was tolerating clear liquids and solid food orally. At home, she reported that she is asymptomatic and has resumed regular activities of daily living within days of discharge. The following morning, the patient's pain score was 0 / 10 and she will be followed closely in our outpatient clinic. The next morning, we take blood samples for complete blood count (CBC), C-reactive protein (CRP), and procalcitonin.

Its not perfect, but its working towards that. 

GUI
The web interface has controls for temperature (lower = more conservative, use 0.3 for clinical stuff), max length (how much text to generate), and strategy (greedy is safe, sampling is creative, beam search is balanced). Hit the "Conservative" preset for the most clinically appropriate reports.
You can run this from app.py


