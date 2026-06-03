Reading between the :

E-mail Reconstruction from Redacted

Dutch FOIA Disclosures

A Master Thesis by Joris Wechsler

Student Number: 15471896

Supervised by Jaap Kamps

Submission date (draft): 19.05.2026

Word Count: 10119

MA Cultural Data & AI

University of Amsterdam

# Table of Contents

[1 Introduction [3](#notes-for-mr.-kamps)](#notes-for-mr.-kamps)

[2 Related Work [6](#related-work)](#related-work)

[2.1 FOIA in the transformation towards Open Governments
[6](#foia-in-the-transformation-towards-open-governments)](#foia-in-the-transformation-towards-open-governments)

[2.2 E-mails in the context of FOIA requests
[7](#e-mails-in-the-context-of-foia-requests)](#e-mails-in-the-context-of-foia-requests)

[2.3 E-mail Parsing Techniques
[9](#e-mail-parsing-techniques)](#e-mail-parsing-techniques)

[2.3.1 Effects of redaction on machine learning
[10](#effects-of-redaction-on-machine-learning)](#effects-of-redaction-on-machine-learning)

[2.3.2 Rule-based Baseline
[11](#rule-based-baseline)](#rule-based-baseline)

[2.3.3 Fine-tuned token classification
[11](#fine-tuned-token-classification)](#fine-tuned-token-classification)

[2.3.4 LLM/VLM zero-shot [12](#llmvlm-zero-shot)](#llmvlm-zero-shot)

[3 Methodology [13](#methodology)](#methodology)

[3.1 Dataset creation [13](#dataset-creation)](#dataset-creation)

[3.2 Pre-processing and ground truth establishment
[14](#pre-processing-and-ground-truth-establishment)](#pre-processing-and-ground-truth-establishment)

[3.3 Model Selection and Configuration
[17](#model-selection-and-configuration)](#model-selection-and-configuration)

[3.3.1 BERT Model Selection and Configuration
[17](#bert-model-selection-and-configuration)](#bert-model-selection-and-configuration)

[3.3.2 Vision-Language Model (VLMs) Selection and Configuration
[20](#vision-language-model-vlms-selection-and-configuration)](#vision-language-model-vlms-selection-and-configuration)

[3.3.3 Hardware and Software Configuration
[22](#hardware-and-software-configuration)](#hardware-and-software-configuration)

[3.4 Evaluation Metrics [23](#evaluation-metrics)](#evaluation-metrics)

[4 Results [25](#results)](#results)

[4.1 Stage 1: E-mail segmentation
[25](#stage-1-e-mail-segmentation)](#stage-1-e-mail-segmentation)

[4.2 Stage 2: E-mail field extraction
[26](#stage-2-e-mail-field-extraction)](#stage-2-e-mail-field-extraction)

[5 Discussion [29](#discussion)](#discussion)

[6 Conclusion [33](#conclusion)](#conclusion)

[7 References [35](#references)](#references)

# <span class="mark">NOTES for Mr. Kamps:</span>

Dear Mr. Kamps,

I hope this thesis finds you well! While I did my best to present a
near-finished version of my thesis, I could not quite manage to do
everything that I would like. Therefore, I would love to hear your
thoughts on things that I think could still improve my thesis in the
upcoming four weeks:

1)  Increase the dataset size and annotate manually without
    pre-annotations. I think the biggest weaknesses of my thesis
    currently is the relatively small dataset (8 dossiers only which
    presents a situation in the e-mail segmentation split that I have 6
    dossiers for training and only 2 for testing all three approaches).
    I feel that a wider variety of dossiers in both train and test would
    make my results more robust. The pre-annotations also leave some
    potential annotation bias that I feel is hard to shake off. Even
    though I did my best to evaluate how I could limit this bias it
    still feels like something that decreases the quality of my research

    1.  This is indeed small, but nonetheless a solid first pilot study.
        So we \
        can deal with any size and explain the limitations very
        carefully. This \
        is best done in the spirit of a half-full glass, with future
        research \
        that clearly shares the construction details so that it can be
        expanded. \
        Adding a few more may be possible and valuable (both in variety
        and \
        quality — think of adding some “ideal” cases where your approach
        works \
        very well). Traditional ML relies heavily on training, whereas
        recent \
        prompt models rely less (so you could divide it differently). 
        For \
        annotation basis, we can have a small sample (1 dossier)
        annotated by a \
        different person and quantify the inter-rater agreement.  The
        selection \
        bias is another issue, but the chosen dossiers are real but
        perhaps not \
        representative (so a little careful to generalize, but this is \
        impossible to avoid).

2)  Maybe an addition to the thesis is to create a modular pipeline of
    sorts that first segments the emails and extract e-mail header
    fields using regex and then uses a VLM/LLM to clean up the headers
    or to reconstruct the email threads? So that we use a ‘best of both
    worlds’ approach to make e-mails more usable Have not really touched
    upon e-mail thread reconstruction yet but I am also not sure if that
    would add much to the current research question

    1.  This is an open problem, and that some extraction problems can
        be \
        solved by a cheap and transparent regex is a positive outcome.
        Just \
        change the framing and order a bit so this strength comes out
        (still \
        good to try alternative and look at success/failure differences
        — maybe \
        a harder case goes better with a more expensive model).
        Identifying \
        remaining errors (threads/headers in email bodies) is a
        strength \
        already, clear ideas on how to address these in future research
        another, \
        and solving this would be another, next thesis…

3)  I do think my discussion section could use some more weight, so any
    feedback on the discussions section specifically is highly
    appreciated!

    1.  Agree that more is possible here. Would be nice to frame the
        thesis \
        around all your literature — and the general accessibility
        problem — and \
        perhaps treat the email as a concrete case study of a very \
        difficult-to-process-and-understand type of disclosed
        information. Then \
        you can derive the main findings from the case study (overall,
        the \
        performance is very high), and draw general conclusions on how
        this \
        helps understanding disclosed information (both in terms of \
        transparency, but also in terms of speed and ease of processing
        — here \
        you could cite the ERP paper, which you can add as an appendix
        to the \
        thesis).  In this way, you have two levels — the literature
        and \
        motivation, and the case study and outcome — and score two
        contributions for the price of one.

4)  Right now I do not refer so much to the previous work we did during
    our ERP project, which I found was a bit hard to weave into this
    thesis (except mentioning that the project preceded this thesis and
    formed the basis of the problem statement).

5)  Right now my word count is around 10 000, which is still roughly 5
    000 under what the minimum is according to the Cultural Data and AI
    Canvas page. I do however also agree with you that longer is not
    always more desirable so I would love to know whether the length is
    okay or whether I need to add more to flesh it out in certain areas.

    1.  No worries. All theses tend to grow, and we can simply count
        the \
        entire PDF. For appendices with details to make it reproducible,
        we can \
        simply include them in the count (these are a key part of the
        thesis). \
        Also, see the point on adding the paper as another appendix.

Thank you for taking the time to read my thesis draft and I look forward
to hearing your feedback! With best regards,

Joris

# Introduction

When the *Volkskrant* published a detailed reconstruction of the
‘mondkapjesdeal’ political scandal that developed during the COVID-19
pandemic, it revealed an unprecedented intimate look inside government
decision making at the highest level through texts and e-mails
(Hendrickx & Kreling, 2022). What was communicated and when became the
groundwork to uncover how one entrepreneur could close a deal directly
with the Minister of Health for a large order of face masks which
resulted in multiple ongoing lawsuits. This case however does not stand
on its own. Investigative journalists relied heavily on government texts
and e-mails to uncover a potential conflict of interest surrounding
Schiphol Airport (Altena, 2020), or where e-mails revealed public
servants advising against their own internal research in a
childcare-benefit scandal (Strop, 2020). The content of these messages
details the day-to-day conversations and decisions that lie behind some
of the largest scandals in recent Dutch political history that cannot be
found in formal letters or memos. As the main form of written
communication, e-mails thus present a crucial source of democratic
accountability for scholars, journalists and citizens.

The reason why these messages are an available source in the first place
is the “Wet open overheid” (Woo), the primary Freedom of Information
Access (FOIA) law in the Netherlands. This law gives the government the
duty to publish certain information proactively, and to publicize
government information when specifically requested. Though there are
still certain grounds upon which information is not shared when this may
result in risks for the government or confidential information, the Woo
forms the foundation of transparency and accountability of governmental
affairs in the Netherlands (Honee & Drahmann, 2024). However there are
few guidelines as to how this information is ought to be released, with
little to no requirements for metadata publication or forms of
publication (Ministerie van Binnenlandse Zaken en Koninkrijksrelaties,
2024). The archival workflow of Woo-dossiers that have been published
seem to favor single pdf’s that contain one stream of all the released
documents after each other, mixing different types of documents with
each other without clear boundaries. This makes structuring, sorting and
interpreting this information a strenuous task, especially for e-mails
whose metadata is crucial for structuring correspondence and temporal
ordering. Although scholars have investigated solutions to splitting
these streams into their distinctive documents, e-mails have generally
not been treated as an unit of study (Ates, 2025; Van Heusden et al.,
2024). Without the metadata of these e-mails, the information remains
locked inside these documents inaccessible to key stakeholders and
end-users.

Simultaneously the archive of e-mails published by the government keeps
growing. While it is not known how many e-mails have been released by
the government in total, the release of COVID-19 related Woo-requests
gives an indication of the sheer size of e-mail correspondence that
enter the public archive. From the 322 990 documents that were
published, more than 30% of documents are e-mails (OpenVWS, 2026).
Still, researchers raise concerns that even though the quantity of open
government data grows, the usability of this data is seriously impeded
by the structural reality of its release (Ruijer, Grimmelikhuijsen, et
al., 2020). When these e-mails cannot be sorted, searched or
contextualized into their original threads, the value that can be
extracted from these records is minimized.

The untapped potential of e-mail archives by the Dutch government was
also recognized by the author’s work at the ICAI Open Gov Lab, where
research was conducted to systematically segment, categorize, and parse
metadata from documents within Woo-dossiers (Terentieva et al., 2026).
However, the practical usability of the researched tool was severely
limited by the quality of extracted e-mail (meta)data and thread
reconstruction. While a Visual Language Model (VLM) based model showed
promising results in segmenting documents, multiple e-mails were
generally treated as a single document and thus the unit of analysis. In
addition, redaction bars and varying page layouts lead to a severe loss
in data quality which impeded the usability of the e-mail extraction
function. Together with hallucinations that VLMs suffer from, the
VLM-based pipeline was not sufficiently accurate and held critical
limitations considering the precarity of these documents and conclusions
that users draw from it. However, considering the relevance for
journalists and citizens to be able to process and analyze e-mails
released by the government on a more efficient scale calls for further
inquiry into how e-mails released by the Dutch government can be
extracted and reconstructed.

Therefore, this thesis aims to answer the following research question:

**To what extent can machine learning methods reconstruct e-mails from
Dutch Woo-dossiers into structured, navigable records, and how could
structural improvements in the release of such documents facilitate
extraction and usability?**

The underlying tasks of this question are as follows:

- Reconstructing e-mail data by segmenting individual e-mails from their
  original PDF streams and extracting all header metadata

- Build a dataset specifically for e-mail extraction on Dutch
  Woo-documents to evaluate the different machine learning methods

- Evaluate feasibility and usability for Woo-stakeholders based on
  computing requirements, usability and scalability

# Related Work

## FOIA in the transformation towards Open Governments

Open government has been one of the major shifts of government
administration in the past 20 years, focusing on citizen participation
through open government data (OGD) policies, and harnessing technology
to increase accessibility and usability of government data (Moon, 2020).
This ‘new open government’ focuses on proactive open data sharing rather
than Freedom-of-Information-Access (FOIA) policies, values interactivity
and participation over accountability and sees citizens as co-producers
rather than mere information receivers (Moon, 2020). OGD is also
considered a broader umbrella of initiatives that center around a more
expansive set of values compared to FOIA and carries more ambiguity in
its agenda (Berliner et al., 2019). While the expanded scope of
government transparency can be hailed as a positive outcome, this shift
is not without consequences for the FOIA community.

Some have argued that OGD introduces the end of FOIA laws (Noveck,
2016). The increase in public and political focus on OGD has been
suggested to displace the goals, priorities and attention from FOIA over
to OGD, which is favored by politicians that see FOIA as more
politically inconvenient (Berliner et al., 2019). Sharing of government
data has a highly strategic component which creates an incentive for
public bodies to refrain from publishing politically sensitive data or
information that undermines their workings (Ruijer, Détienne, et al.,
2020). Political resistance against the Woo is also growing in the
Netherlands, where OGD is hailed as the successor to alleviate the
workload that Woo presents today (Hartholt, 2025; Schohaus, 2026).
Therefore, the Woo, while not without its limitations, holds a unique
possibility to foster transparency and accountability in a more targeted
and concrete manner than OGD.

Research by Cucciniello et al. (2017) suggests that core values such as
accountability, trust and legitimacy are not guaranteed outcomes of
transparency initiatives. The legal requirements of FOIA laws
insufficiently consider citizen’s preferences for data which diminish
the usability and therefore the effective transparency of such laws
(Cucciniello & Nasi, 2014). While the government thus may claim
transparency by releasing government documents, the lack of usability
may significantly decrease to what extent citizens, journalists or
researchers can harness this data. Scholars have identified several
examples where governments comply to FOIA requests while diminishing the
usability for citizens, such as reducing responsiveness on larger
requests (Spáč et al., 2025), providing data in difficult-to-use formats
(Luscombe & Walby, 2017) or delaying release of documents beyond the
legal requirement (Mario & Kilty, 2025). These processes contradict the
essence of ‘transparency-by-design’ which offers a differentiated view
that treats transparency as a factor in each phase of system design
(Janssen et al., 2017). When information is not easily usable by its
recipients, it severely limits the ability of FOIA laws to foster
accountability. Strengthening FOIA archives to enhance usability,
interpretability and retrievability thus appear not to be in line with
the political agenda towards OGD and therefore requires solutions from
academia, journalism and citizens to ensure accountability within the
existing legal and structural reality.

## E-mails in the context of FOIA requests

As much as FOIA request is a legal process, Luscombe and Walby (2017)
propose a recognition of FOIA-requests as a live archive, a production
and retention of government records much like traditional archives. What
sets it apart from a traditional archive is that the content still
‘lives’ within the governmental organization and is subject to the act
of archiving of government officials which turns accountability into a
process of everyday government workers (Luscombe & Walby, 2017).
Similarly, the issues pertaining FOIA collections are echoed in archival
studies as well, such as accessibility issues of digital resources, lack
of interface possibilities but also the possibilities that machine
learning and AI can present for possible solutions (Baron, 2025;
Jaillant, 2022). In particular, there is currently no academic consensus
on how to preserve e-mails best (Allegrezza, 2022).

As much of government communication occurs via e-mail, archives of these
messages have long been considered valuable evidence for government
accountability (Johnston et al., 2019) and therefore also FOIA requests
where emails appear. The controversy surrounding Hillary Clinton’s use
of private e-mail servers and the Jeffrey Epstein e-mail release are
just two recent examples of cases where e-mails formed the foundation of
FOIA releases (Leopold, 2026; Millhiser, 2019). In both these cases, the
role of e-mail exchanges and the content thereof are both of highly
political and legal importance in U.S. politics. However, the way in
which these e-mails are released also point to a deeper understanding of
how FOIA releases can enhance transparency. The Clinton e-mails released
by the Department of State were released as machine-readable PDF’s with
separate metadata fields on sender, receiver and date. While not all
metadata fields are complete and suffers from mistakes, the disclosure
shows what downstream possibilities are enabled by providing structure
and context to these files. The structural affordances of such a release
allowed for the creation of visualization and analysis tools by
journalists and academics, such as the searchable database by WikiLeaks
and the Washington Post or analysis tools by New York University
(Sigdyal, 2016; WikiLeaks, 2026). Still, data quality issues pertain as
multiple e-mails are concatenated in one PDF, OCR errors persist and
redactions remove linguistic coherence (De Felice & Garretson, 2018).
The Epstein e-mails, totaling 3.5 million files released by January 2026
and thus a much larger release than the Clinton e-mails, were published
quite differently (U.S. Department of Justice, 2026). The release was
rather unstructured with an absence of any metadata, releases in
different file formats and difficult to use release platforms,
highlighting that e-mail parsing still remains an issue today (Hannah,
2026). Still, various attempts at structuring this data have been
observed, such as the Jmail platform that uses a commercial, LLM-based
document parsing technology to re-introduce structure to these e-mails
and allow for indexability and searchability (Banikarim, 2026). Still,
some scholars express concern regarding these generative technologies
catered towards consumers that want to investigate such FOIA requests
(Hannah, 2026).

Still, there exists no academic dataset for e-mail parsing which limits
the possibilities to validate such commercial document parsers and
removes the possibility to find alternative ways to parse e-mails more
effectively. In the Netherlands, all documents released under the Woo
are centrally published online and accessible for anyone, creating a
platform with a wealth of information. The documents are released as
part of their respective Woo-dossier in a (if applicable) redacted form
in a pdf stream, yet with no metadata fields the e-mails are not easily
organizable, searchable and thus usable.

## E-mail Parsing Techniques

The objective of this research touches on two streams of document
understanding research, namely *document segmentation and document
parsing.* While the primary task involves splitting concatenated e-mails
into their respective units, the secondary task provides structure to
those units for further analysis. While these tasks come from distinct
research lineages, in practice both are needed to increase the usability
of their original form and to achieve a structured output of this data.

The proposed task of segmenting individual e-mails from a stream of
e-mails sits in between multiple research disciplines that have been
applied to similar contexts and can inform possibilities of evaluation
and methodological construction moving forward. In general, the unit of
analysis informs the research context and application possibilities for
this research. *Text segmentation* is used as a broader research
umbrella and is interested in the splitting of text into sub-units of
interest, commonly into words, sentences or topics (Pak & Teh, 2018).
Slightly larger units is the focal point of *Email Zoning,* which splits
individual e-mail text into ‘functional zones’ such as signatures or
thread reconstruction. Previous work to analyze Woo-dossier data center
mostly around *Page Stream Segmentation* (PSS) to extract individual
documents from PDF streams as a page-wise binary classification task
(Ates, 2025; Van Heusden et al., 2024). However, no current research has
focused on the individual e-mails within those documents as the unit of
study, even though many Woo-dossiers consist of large swaths of e-mails
that do not strictly follow page breaks. Still, lessons from the PSS
literature can inform the choice of methodology to translate to the task
of e-mail span detection, in particular machine learning techniques and
the benefits of multimodal ensembles (Van Heusden et al., 2024).
Combining the binary classification task of PSS with parsed text as the
raw data combines these two approaches to fit the proposed objective.

Once the unit of analysis has been established through segmentation, the
second objective of this research is the structuring of key e-mail data,
namely the e-mail headers, for further analysis. *Document parsing*
refers to the structuring information contained in previously
unstructured documents, with respect to the structure of various
document elements such as tables, text and images (Q. Zhang et al.,
2024). While the main focus of this field lies in challenging document
layouts such as financial reports, textbooks and newspaper layouts,
e-mails have mostly been absent in benchmarks (Ouyang et al., 2025).
This may be due to the generally clear structure that e-mails follow;
however, this often assumes unredacted and clean data which in the
practical reality of FOIA requests rarely is the case.

In recent years, two main methodological streams have emerged to parse
documents at scale: modular pipeline systems and end-to-end VLM-based
models (Q. Zhang et al., 2024). While modular pipelines can be adapted
and follow a sequential approach with elements such as layout analysis
and content extraction, VLM-based pipelines provide an end-to-end
parsing which combines textual, visual and structural information (Q.
Zhang et al., 2024). When specifically looking at text classification as
a sub-task of document parsing, evidence shows that ‘older’ fine-tuned
text classification models still outperform generative models (Vajjala &
Shimangaud, 2025). While LLM’s have received significant attention for
this task, traditional text classification models such as BERT are more
suited for texts that follow more regular patterns, which e-mails
arguably fall under (J. Zhang et al., 2025). Still, VLMs have advanced
significantly over the past years and offer a solution that is flexible,
easy-to-use and end-to-end solution that may prove beneficial for
practitioners. Therefore, both methodological streams are included as an
evaluation for the e-mail segmentation and parsing, as well as a
baseline regex expression pipeline for comparison.

### Effects of redaction on machine learning

Based on the legal grounds that the Woo is based on, Woo-released
documents include redactions to protect personal information from
employees and sensitive information. While the Dutch government does not
share information on their redaction practices, it can be assumed that
this is done manually using a redaction software.

In particular for natural language processing (NLP) tasks, redaction
bars present a technical challenge because they make documents more
difficult to read and understand and therefore limit the abilities of
text classification models. In the context of Dutch Woo-requested
documents, it has been observed that documents are provided as scanned
documents, where Optical Character Recognition (OCR) is needed to
extract any text (van Heusden et al., 2025a). In other cases, the raw
text that is extracted from these PDF’s show signs of OCR noise,
particularly around redaction bars, which leads to the assumption that
some of these documents are stripped from their original text which is
later added again through OCR. Bland et al. (2022) argue that this
process, rasterizing PDF’s into images, can also be considered a way to
increase the security of redactions. As van Strien et al. (2020)
demonstrated, OCR noise and errors can significantly decrease downstream
NLP tasks, particularly on segmentation tasks. When Jiang et al. (2022)
conducted an experiment by comparing BERT’s abilities to create semantic
relations in scholarly articles, they found that the performance of
different BERT models drop by 10%-20% when faced with noisy text. Noisy
text therefore presents a particular challenge for models that obtain
their context from the semantic relation between words. Current research
on redacted text and machine learning focus mostly on how to automate
the redaction of sensitive text (e.g. Jeon et al., 2025), though no
research has been found that measures the impact of redactions on
classification tasks.

### Rule-based Baseline

The first approach consists of a series of regular expression rules to
extract the most common header keywords in e-mails which serves as the
baseline to which the other techniques are compared. As e-mail fields
tend to have identical lines such as From:, To:, Subject:, regular
expressions lend themselves well to extract data from e-mail headers
(Mahlawi & Sasi, 2017). Therefore, a combination of these expressions
would constitute the start of a new e-mail, whereas the metadata
(sender, receiver, subject and date) would consist of the text in
between those expressions. While computationally inexpensive, the risk
of redactions and text quality issues might decrease the results.
Redaction codes that are commonly found in Woo documents that may read
as text can also be recognized as such using regex expressions (van
Heusden et al., 2023).

### Fine-tuned token classification

The second approach uses token classification as a method of e-mail
segmentation and metadata extraction. By using a BIO (Beginning, Inside,
Outside) tagging scheme, tokens can be divided into meaningful named
entities, in this case e-mail header categories, which structures the
e-mail data (Jurafsky & Martin, 2026). An annotated dataset is then used
to fine-tune a pre-trained language model to introduce and improve on
domain-specific data.

This technique may overcome structural degradation from heavy redactions
that the rule-based baseline suffers from and uses more semantic context
of the text to classify tokens. At the same time, it is architecturally
extractive and therefore do not suffer from hallucinations like VLMs.
This is in particular relevant for stakeholders of Woo dossiers such as
lawyers and journalists that depends on results that are strictly
aligned with the original text.

### LLM/VLM zero-shot

The third approach explores the possibilities offered by VLMs as an
end-to-end solution to e-mail segmentation and parsing, as promising
results have emerged in recent years using this technique (Ates, 2025).
The simultaneous processing of visual, textual and structural
information may prove beneficial when tasked with heterogenous data
structures such as Woo documents. At the same time, VLMs combines the
aforementioned subtasks into one unified approach that eliminates the
need for task-specific models and complex modular pipelines. In
particular, multi-modal approaches have proven beneficial for detecting
text redaction and layout awareness (van Heusden et al., 2025a). The
thesis will follow a dual approach where a text-based and a text- and
vision-based run are compared to each other to analyze the added benefit
of adding the vision modality to e-mail segmentation and e-mail field
extraction.

### 

# Methodology

## Dataset creation

As discussed previously, the task of e-mail segmentation and e-mail
field extraction poses a unique, yet important task within the realm of
FOIA document release. While datasets have been created using Dutch
Woo-documents, these mainly focused on PSS (Van Heusden et al., 2024)
and redaction bar detection (van Heusden et al., 2025b), which do not
translate to the task at hand. Subsequently, no structured dataset was
found that contains the challenges faced by FOIA users, namely
concatenation of documents and varying degrees and forms of redaction.
Therefore, this thesis contributes to the ongoing academic endeavor for
increasing accessibility to FOIA archives by creating a dataset to
evaluate different computational techniques for e-mail segmentation and
metadata parsing.

In the context of Dutch FOIA requests, Woo-requested documents are
publicly available on the open.overheid.nl website from a wide variety
of government bodies. Documents vary strongly in length, types of
documents included and structure depending on the initial request and
the release decision by the responsible ministry. The documents are
nested inside PDF page streams and therefore creating an accurate
overview of e-mails within Woo requests remains challenging. Figure 1
shows a selection of structural differences within these documents.

![](media/image4.png)

**Figure 1:** Variety of document structures and redaction codes

In order to construct a dataset that reflects the variety of Woo
documents, the following eligibility criteria were set for selecting
relevant dossiers:

- **Variety of ministries**: In order to reflect the different workflows
  and software used by different ministries, the dataset aims to
  represent a variety of ministries as a selection criteria. It is
  assumed that workflows within ministries are generally identical and
  therefore would not yield additional insights.

- **Recency**: As archival practices develop and improve over time, the
  decision was made to include only dossiers from within the past two
  years. The aim is to reflect the current practices of Woo workflows as
  government bodies may have introduced improved techniques to release
  documents.

- **E-mail foundation**: As the focal point of this thesis surrounds
  e-mails, dossiers were only eligible that consisted for a large
  fraction of e-mails.

- **Variety of document-specific structures:** The dataset aims to
  incorporate a wide variety of redaction bars, e-mail formats and
  internal/external communication. As van Heusden et al. (2023)
  highlighted, a variety is needed in order to reflect the reality of
  Woo dossiers.

Based on the selection criteria, eight dossiers were manually selected
to form the dataset for training and testing, and after removing all
non-email documents, 543 pages of e-mail data remained. The shortest
dossier was 12 pages while the longest dossier counted 126 pages, with
an average of 68 pages per dossier. All documents were in Dutch, however
based on the user’s settings the e-mail fields were in either English or
Dutch. The dataset represents eight different ministries, including the
three ministries with most Woo requests per Woozm.nl. These documents
were then used to establish the ground truth for further model
fine-tuning and testing.

## Pre-processing and ground truth establishment

After removing all non-email documents from the dossiers, the pdf files
were imported into Visual Code for pre-processing. The following steps
were taken to process the pdf’s before annotating the ground truth:

- **Native text extraction:** The first step was to ensure that native
  text was present in the pdf files to enable the segmentation using the
  text-based methods. Extraction of native text was conducted using
  pdfplumber version 0.11.9 which resulted in readable text for six out
  of the eight pdf files. During this step it was noted that even some
  dossiers that were machine readable included OCR artefacts. The
  remaining two dossiers produced unreadable text, likely due to
  corrupted character encoding. Therefore, an OCR extraction was
  necessary for the remaining two dossiers.

- **OCR text extraction:** The remaining two dossiers were rendered as
  .png images and run through Tesseract 5.5.2 to extract a text layer.
  This OCR engine was used as it offers a balance between cost and
  resource-effectiveness, Dutch language support and reliability for
  typed text.

- **No text cleaning:** A deliberate choice to forego text cleaning was
  made as the dataset is intended to showcase real-world data and such
  artefacts thus also reflect realistic cases of FOIA accessibility.

- **Pre-annotation**: Due to the highly structured nature of e-mail
  fields, a pre-annotation based on regular expressions was conducted to
  increase the speed of annotation. Pre-annotations have been recognized
  in various fields as time saving without introducing bias (Lingren et
  al., 2014). The result section will further reflect on this approach
  to limit pre-annotation biases introduced by system (Mikulová et al.,
  2023).

  - **E-mail segmentation**: In order to segment e-mails, an e-mail
    start was triggered though a combination of either a ‘From:’ or
    ‘To:’ field with a ‘Subject:’ field within the next 500 characters.
    Potential duplicates were removed and e-mail spans were defined to
    continue until the start of the next e-mail.

  - **Field extraction:** For e-mail field extraction, keywords were
    searched (i.e. ‘CC:’, ‘Verzonden:’, ‘Subject:’ etc.) at the start of
    each line, after which the value was defined as the characters after
    the colon on the same line (multiple lines were allowed for sender
    and recipient fields only).

To create the pre-annotations using a regex and subsequently treat a
regex function as a method of comparison may rightfully call into
question whether an annotation bias will be introduced with this method.
However, this section will outline the difference between the created
annotation and the manually corrected ground truth, while in chapter 3.4
an evaluation metric is discussed to decrease the effect of a potential
annotation bias. The impact of this will be further discussed in the
discussions chapter. The manual annotation, which included verification
and correction of the pre-annotation candidates, was conducted in a
two-phase approach using Label Studio.

The first round only pre-filled the e-mail segmentation candidates from
the pre-annotation for manual verification. As the regular expression
used strict signals to segment e-mails, a tendency of undersplitting was
identified which was corrected. The second round split all e-mails
according to the manual annotation and pre-filled the candidate e-mail
fields from the pre-annotation and were manually corrected. The results
of the manual annotation show significant adjustments made to the
pre-annotation, particularly for the first stage, as can be seen in
Table 1. The second stage required less adjustments as the fields
followed highly structured and anticipated regular expressions. Due to
the strict parameters of the regex function, no prediction was deleted
altogether. Table 1 shows the results from the pre-annotation compared
to the manual annotation.

**Table 1**

*Pre-annotations and manual annotations*

<table style="width:99%;">
<colgroup>
<col style="width: 16%" />
<col style="width: 1%" />
<col style="width: 16%" />
<col style="width: 14%" />
<col style="width: 16%" />
<col style="width: 16%" />
<col style="width: 16%" />
</colgroup>
<thead>
<tr>
<th></th>
<th colspan="2" style="text-align: center;"><strong>Total accepted
pre-annotations</strong></th>
<th style="text-align: center;"><strong>Total adjustments</strong></th>
<th style="text-align: center;"><strong>Modified
pre-annotations</strong></th>
<th style="text-align: center;"><strong>Added annotations</strong></th>
<th style="text-align: center;"><strong>Total manual
annotations</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="2">Stage 1: e-mail segmentation</td>
<td style="text-align: center;"><p>486</p>
<p>(56.9%)</p></td>
<td style="text-align: center;"><p>368</p>
<p>(43.1%)</p></td>
<td style="text-align: center;"><p>277</p>
<p>(32.4%)</p></td>
<td style="text-align: center;"><p>91</p>
<p>(10.7%)</p></td>
<td style="text-align: center;">854</td>
</tr>
<tr>
<td colspan="2">Stage 2: e-mail field extraction</td>
<td style="text-align: center;"><p>3305</p>
<p>(85.4%)</p></td>
<td style="text-align: center;"><p>565</p>
<p>(14.6%)</p></td>
<td style="text-align: center;"><p>410</p>
<p>(10.6%)</p></td>
<td style="text-align: center;"><p>155</p>
<p>(4.0%)</p></td>
<td style="text-align: center;">3870</td>
</tr>
</tbody>
</table>

*Note: Total adjustments is the sum of modified pre-annotations and
added annotations.*

Subsequently, both ground truth datasets (e-mail segmentation and e-mail
fields) were split into an 80/20 training and testing set, though their
unit of analysis differed which had consequences for the train/test
split. For stage one, splitting occurred at the dossier level, which
resulted in six dossiers for training and two dossiers for testing, with
seed set at 42. The training set consisted of 612 training e-mails and
242 test e-mails. For stage two, the splits were stratified by dossier
to ensure all dossiers were included in both training and testing. The
stage two training set consisted of 682 training e-mails and 171 test
e-mails, with seed set at 42.

## Model Selection and Configuration

### BERT Model Selection and Configuration

In the field of natural language processing, transformer-encoder models
such as BERT have become widespread due to their success at task
completion (Gardazi et al., 2025). Part of its strong capabilities can
be explained by its bidirectional nature, incorporating context from
left and right. Since its inception, many variations of BERT have been
introduced that address either specific limitations, or improved
performance based on the intended task. These improvements occur in the
architecture or in the pre-training data of the model. In general,
variations of BERT models occur either through knowledge enrichment
(e.g. LIBBERT), multilingual/language-specific training data (e.g.
RobBERT, mBERT), domain-specificity (e.g. BioBERT) or model compression
(e.g. distilBERT) (Qiu et al., 2020). Therefore, choosing a variant that
more closely aligns to the task at hand can increase the performance
significantly. With Dutch FOIA requested documents, the options for
variants are a structure-specific model such as EmailBERT which was
trained on the English Enron e-mail dataset (Jin, 2025), or a Dutch
language-specific model such as RobBERT or BERTje (Delobelle et al.,
2020). Multi-lingual models were disregarded at this stage as
monolinguistic models have shown to outperform multi-lingual models,
such as de Vries et al. (2019). However, experiments have shown that a
mismatch between tokenizer language and target text language can lead to
model failure as the tokens are arbitrarily set (Rust et al., 2020). As
most of the dataset consists of Dutch text, using a Dutch-specific model
was expected to result in better performance.

The two major Dutch BERT models, BERTje and RobBERT differ both in their
training foundation as well as their training data. Whereas BERTje was
built on the original BERT architecture, RobBERT was trained on RoBERTa,
an optimized version of BERT with improved training procedures
(Delobelle et al., 2020; Liu et al., 2019). The second difference lies
in the training data, both in size and in source material. BERTje was
trained on a corpora of Dutch text from high quality sources, such as
historical fiction novels, newspaper articles and the SoNaR-500 corpus
(de Vries et al., 2019). The training data used for the original RobBERT
on the other hand was roughly three times as large and was based on the
Dutch OSCAR corpus, obtained from crawled websites (Delobelle et al.,
2020). An updated version of RobBERT in 2023 introduced new vocabulary
and better quality filtering, and used a Tik-to-Tok strategy to
efficiently train the updated model (Delobelle & Remy, 2024). While both
a base and large model exist of the updated RobBERT model, due to
resource constraints the base model was the more suitable option. Due to
the more modern architecture of RobBERT, the vastly larger training data
used, and the training data consisting mostly of internet sources which
may show closer similarity with the writing style in e-mails, the
RobBERT-2023 base model was selected (DTAI KU Leuven, 2023).

In order to achieve the best results for each stage, the appropriate
output head requires some consideration as the unit of classification
slightly differs between the two. While BIO-token classification was
initially tested for the task of e-mail segmentation, the model over
segmented e-mail starts considerably. The root cause was suspected to be
the 512 token limit that is intrinsic to the BERT models and can
seriously impede performance (Sebők et al., 2025). Woo dossier e-mails
often exceed this length, thus leaving a large part of predictions to
occur without an e-mail header in its window even with stride set at
128. Simultaneously, the BIO-token classification forces the model to
produce a prediction for each token, which in our task is not required.
These limitations motivated a reconsideration of the classification
layer to a line-level binary classification task. As e-mails only start
at the beginning of a new line, the model was thus tasked to classify
each line as an e-mail start or not. The input window was expanded by
two context lines on each side and concatenated with the target line.

For the e-mail field extraction, a BIO-token classification layer was
employed that assigns each token in an e-mail to either an O-field
(outside token), B-field (beginning token) or I-field (inside token) for
each of the six field types: FROM, TO, CC, DATE, SUBJECT, ATTACHMENT.
Utilizing the beginning and inside classification allows the
reconstruction of multi-token spans, for each field to compare to the
ground truth. Token classification was selected as it is widely used for
Name Entity Recognition (NER) tasks which shares a similar task
description and output scheme as this study.

A common problem that occurs with token classification tasks is an
extreme data imbalance, where a majority of tokens are labelled as the
(O) outside class (Nemoto et al., 2025). In this study this problem is
further exacerbated by the multitude of classes in our classification
scheme. By capturing six field types this totals to 13 labels, meaning
the tokens that are already a minority class are further fragmented.
This imbalance leads to a tendency for misclassifying tokens as O-class,
which leads to poor predictive capabilities for the minority classes
(Nemoto et al., 2025). Focal Loss is a widely used fine-tuning method to
counteract this tendency, by down-weighing penalties for high-confidence
predictions and thus forces the model to focus on more low-confidence
predictions, which constitute the minority class (Lin et al., 2018; van
Leeuwen et al., 2025). In an initial trial-training, the RobBERT model
showed the strong class imbalance that was predicted. While some classes
had relative strong representation in the ground truth, such as I-TO
with 9.75% of tokens, no B-class token appeared more frequently than
0.41% of the total tokens. The B-class is a crucial token as a sequence
span cannot be calculated without it. In the trial-test, the effect of
an underprediction of B-class tokens showed significant effects. Three
e-mail fields (CC, SUBJECT and ATTACHMENT) collapsed completely and
showed no predictions at all. While the model predicted I-classes for
these fields, no B-class were preceded by these fields and thus no
sequence could be established. The focal loss approached failed to
direct sufficient focus between classes, in particular the B-classes
that occur least frequently out of all classes yet are of great
importance. Therefore, a smoothed inverse square-root class weight was
introduced to balance the between-class frequency issue. By using the
square-root, extreme weight ratios are avoided that can occur in raw
inverse frequency methods. The results of the weighted RobBERT model
showed a substantial increase in recall across all e-mail fields, with
an average of 0.213 increase per field. This increase shows that the
model handles class imbalance more effectively and therefore will be
reported alongside the unweighted RobBERT model for transparency.

The line-level classifier for stage 1 was trained with five epochs with
a batch size of 32. Binary cross-entropy with square root inverse
frequency class weights is applied using Focal Loss (γ = 2.0) to address
the imbalance between the imbalance between e-mail start lines and
non-email start lines.

The unweighted token classifier for stage 2 was initially trained for 5
epochs with a batch size of 16. Focal loss at (γ = 2.0) was applied
without class weights. The collapse of the three fields did not warrant
further training with increased epochs; therefore, this number was not
increased.

The weighted token classifier for stage 2 was configured identically to
the unweighted, with three important changes. The α parameter of the
Focal Loss was weighted using square root inverse frequency class
weights. The training ceiling was raised to 15 epochs and early stopping
was introduced with a patience window of three consecutive epochs
without improvement in validation loss. The effective number of epochs
was determined by the early stopping mechanism rather than the ceiling.

### Vision-Language Model (VLMs) Selection and Configuration

The landscape of multimodal VLMs is constantly expanding, with the large
commercial AI companies developing the highest performing models. For
this pipeline, the aim was to compare the fine-tuned token
classification to an out-of-the-box model, readily accessible for
regular users of Woo-requests that does not require any technical
knowledge. OpenAI’s most current model, GPT-5.5 is describes as the
state-of-the-art model, though it also has strong limitations when
applied across different domains (Guruprasad et al., 2025). Still, the
development of VLMs is rapidly evolving and due to its promising
capabilities still makes it an important point of comparison. Rather
than running local VLMs which were not feasible due to resource
constraints, GPT-5.5 was therefore chosen to show the current
capabilities of VLMs in this domain. The GPT-5.5 approach, due to its
end-to-end, zero-shot capabilities which requires no further technical
configuration, represents a relevant practical tool for Woo stakeholders
and users.

For the e-mail segmentation task, the length of the Woo dossiers poses a
practical limitation: the cost of inputting images from hundreds of
pages can rise quickly with frontier models, as well as image size
limits that reduce the ability to upload entire Woo dossiers. Therefore,
this study will take a dual approach that compared the abilities of a
text-only run with a text- and vision-run combined for both e-mail
segmentation and e-mail field extraction.

For e-mail segmentation, the full dossier was submitted once as a single
text prompt with an expanded token ceiling to avoid response truncation
for longer dossiers. The model was prompted to extract all individual
e-mails in the text, with the DATE field as the main anchoring point.
Using the DATE field as the main anchor to detect individual e-mails
rather than the first line FROM was taken due to the heterogeneity of
redaction practices. Most dossiers have fully redacted FROM fields, and
in some cases one redaction bar is used for the entire field. This
results in all FROM fields producing identical field values, which make
it impossible to distinguish from one another. The DATE field is unique
in almost all cases and never redacted, therefore offering a better
anchor to detect individual e-mails. The output in this step is a JSON
array of DATE fields, that are being matched using string search. The
e-mail spans are then constructed by identifying the closest preceding
FROM field and using that position to determine the span from the start
of the e-mail until the start of the next e-mail. In order to evaluate
the contribution of visual information for e-mail segmentation, a second
API call was conducted with both the PDF pages of the e-mails converted
to PNG images alongside the extracted text. Due to upload constraints,
the images were rendered to 72 DPI, lower than the more commonly
accepted 150 DPI. The limitations of this approach will be further
discussed in the Discussions chapter. The model was instructed to use
the page images as a primary source, though the extracted text was used
to follow the same date-anchoring step and span construction logic. The
deliberate omission of employing the VLMs OCR capabilities was made as
the VLM pipeline would likely produce OCR discrepancies compared to the
ground truth which would misleadingly produce a poorer performance.

Similarly to the e-mail segmentation, a text-only and a text- and
vision-based run were conducted to compare the potential benefit of
including visual modality to the prediction. Due to OCR noise, the
instructions for the e-mail field extraction were adapted to produce the
e-mail field values without any text cleaning. In the first iteration,
field values that were not easily interpretable were silently dropped
from the results. Therefore, a heuristics approach to include any
values, regardless of redaction or OCR artefacts was instructed. The
output scheme was a JSON file that lists all the field values and allows
for multiple fields in the CC and ATTACHMENT fields. To evaluate the
vision modality contribution, the vision- and text-based run was
conducted by submitting the extracted e-mail text alongside the
corresponding page images in an API call. Pages were rendered at 150 DPI
as API calls were made per e-mail which circumvented the API size
limits. The prompt structure and output followed the same approach as
the text-only method.

The GPT 5.5 model was pinned to the *gpt-5.5-2026-04-23* release and
temperature was set at zero to increase deterministic outputs.

### Hardware and Software Configuration

All the coding and configurations were developed in Python using VS
Code, with coding assistance from Claude Code. The codebase is
accessible in the following GitHub repository:

*https://github.com/joriswex/woo-email-extraction*

While the regex experiment and the GPT-5.5 API calls were made locally,
BERT training was performed on Kaggle’s cloud infrastructure using
Kaggle Notebooks using an NVIDIA Tesla T4 (1x) with 15GB VRAM.

The inference and evaluation were run locally using an Apple M3 8-core
CPU, 8 GB unified memory on macOS 26.0. All experiments were conducted
in a Python virtual environment with the following core stack:

| **Package**     | **Version** | **Purpose**                               |
|-----------------|:-----------:|-------------------------------------------|
| Python          |   3.13.7    | Runtime                                   |
| PyTorch         |   2.11.0    | Machine learning framework                |
| Transformers    |    5.8.0    | RobBERT model and fine-tuning             |
| Tesseract       |    5.5.2    | OCR for dossiers with missing text layers |
| Datasets        |    4.8.5    | Data management for training              |
| Accelerate      |   1.13.0    | Device placement                          |
| ANLS_star       |    1.01     | Calculation of ANLS metric                |
| Auxiliary tools |     \-      | Pdfplumber, pytesseract, Pillow, openai   |

## Evaluation Metrics

In order to evaluate all three methods with each other, metrics are
needed that use the same unit of analysis. In practice, this means that
BERT tokens are converted back to strings to that our results can be
compared as strings of characters rather than tokens. The comparison is
then primarily centered around precision, recall and the F1 score, which
are widely used metrics for evaluation of string classification tasks
(Keraghel et al., 2024).

The precision metric calculates the total correct predictions by the
model compared to the total predictions made by the model, both true and
false.

``` math
precision = \ \frac{True\ Positives}{True\ Positives + False\ Positives}
```

The recall metric calculates the total correct predictions by the model
compared to the total correct instances in the ground truth.

``` math
recall = \ \frac{True\ Positives}{True\ Positives + False\ Negatives}
```

As both metrics share importance to evaluate a model’s performance, the
F1 score can be calculated that creates a harmonic mean between both
metrics. It calculates a model’s ability to find true positives while
limiting false predictions, both false negatives and false positives.
The F1 score is calculated as follows:

``` math
F1 = 2\  \times \ \frac{precision\  \times \ recall}{precision + recall}
```

The F1 score can, if multiple classes or categories are present, be
calculated as a macro-average across all fields, or as a micro-average
that adds weight to the frequency of predictions per class (Keraghel et
al., 2024). For the first stage, e-mail segmentation, only the macro F1
is available for analysis. For the e-mail field extraction experiment,
both the micro and macro average are therefore relevant for analysis as
not all fields appear in equal frequency.

Specifically for the e-mail segmentation stage, e-mails were considered
to be a true positive when the overlap between the annotated e-mail and
the predicted e-mail share an Intersection over Union (IoU) of at least
0.5 (Van Heusden et al., 2024). This approach allows to measure the
objective of this analysis which is to create structured and navigable
records for downstream analysis, which is satisfied when the e-mail
boundaries are approximately identified. Due to the heavy OCR noise and
inconsistent formatting of Woo documents, precise boundaries are
difficult to draw even for human annotators, therefore some leniency in
the boundary arguably strengthens the evaluation.

While the F1 score is a widely used in academia, scholars have also
noted that OCR noise may unnecessarily penalize approximate matches
rather than the strict true/false binary of F1 scores (Peer et al.,
2025). As the preprocessing of documents showed, Dutch Woo dossiers
suffer from heavy redactions and considerable OCR noise, which can
negatively affect the performance of NLP models. An answer that is
almost correct, offset by a few characters that contain OCR noise, would
still be considered a correct answer as it does improve the field
extraction of e-mails. In order to circumvent this issue, the Average
Normalized Levenshtein Similarity (ANLS) metric calculates the
similarity between the annotated ground truth and the prediction on a
single character basis. It uses a raw similarity threshold of ≥0.5 to
avoid penalizing OCR noise but still remains strict on false predictions
or hallucinations (Peer et al., 2025). The score is only calculated for
fields that have a value present as the score otherwise inflates scores
for fields that appear less frequently such as ATTACHMENTS. In addition,
the ANLS metric serves an important methodological function: lowering
the possible effect of pre-annotation bias introduced by the
pre-annotation method. As the ground truth was established with a regex
function as a suggestion, a strict exact matching of fields would favor
the regex method. However, by introducing the ANLS metric and
pre-processing the outcomes for the F1 score (which include removing
whitespace and upper-case letters), this advantage is counteracted
significantly. Finally, the ground truth is manually checked and
adjusted, thus still presenting a strong baseline to compare all three
methods to. This metric will be calculated and reported alongside the
exact F1 match to compare the difference and analyze the impact of
approximate matches in the dataset.

# Results

The results will be presented in their respective stages as each stage
requires slightly adapted evaluation metrics that are relevant to the
task at hand. To report the F1 score, all strings are normalized by
converting all values to lowercase and collapsing of all whitespaces.
This is especially relevant as the redaction bars introduced irrelevant
white spaces that may inflate errors.

## Stage 1: E-mail segmentation

The e-mail segmentation task involved splitting a concatenated document
of e-mails into their respective individual documents. The span levels
were considered as a true positive when the overlap of the annotated
e-mail span and the predicted e-mail was 50% or larger.

**Table 2**

*Results of e-mail segmentation*

| **Approach**     | **Precision** | **Recall** |  **F1**   | **Predicted e-mails** |
|------------------|:-------------:|:----------:|:---------:|:---------------------:|
| Regex            |     0.926     |   0.922    |   0.924   |          241          |
| RobBERT (token BIO) |     0.370     |   0.943    |   0.531   |          621          |
| GPT-5.5 (text)   |   **0.971**   | **0.971**  | **0.971** |          242          |
| GPT-5.5 (vision+text) |     0.913     |   0.656    |   0.726   |          166          |

The results in Table 2 show that the GPT-5.5 (text) pipeline performed
best with a F1 score of 0.971, closely followed by the regex baseline
with a F1 score of 0.926. The GPT-5.5 (vision+text) model, though very
precise, underpredicted severely leading to an F1 score of 0.726. A more
thorough analysis shows that while one dossier was predicted perfectly,
the other dossier performed very poorly with recall at 0.311. The
RobBERT fine-tuned model falls behind significantly with an F1 score of
0.531. While the recall of the RobBERT model is on par with the other
pipelines at 0.943, it produces low scores for precision at 0.370.
Further analysis of false positives shows the root cause of the drop in
precision: the model predicts beginning of e-mails at a much higher rate
than the other models. While regex almost predicted the correct number
of e-mails, and GPT-5.5 (text) only missed one e-mail, the RobBERT
pipeline predicted significantly more e-mails. Looking at the raw
predictions the RobBERT pipeline predicted multiple e-mail starts within
the same e-mail header. From the 328 false positives recorded, 234
instances predicted a new e-mail start at the ‘Sent:’ line while it was
still part of the previous e-mail. It disregards the standardized
structure of e-mail headers and only looks at text features and labels
provided, which leads to many false predictions.

Since the RobBERT pipeline shows a significant structural limitation in
recognizing the structure of e-mails leading to low precision scores, in
addition a filter was added to disregard predictions that occurred in
close proximity to earlier predictions. This would suppress predictions
made in close proximity to ‘From:’ lines, which generally mark the
beginning of e-mail headers. Predictions made within eight lines of each
other are collapsed into the previous prediction. While this change
presents a post-hoc improvement of the RobBERT pipeline and thus limits
the comparability to the other benchmarks, it does indicate whether
small structural rules could circumvent the bottleneck presented by the
token classification architecture. With the filter, precision increased
significantly to 0.966, while recall dropped slightly to 0.901. Still,
the strong increase in precision lifts the F1 score to 0.932 which
situates the score closer to the other pipelines.

## Stage 2: E-mail field extraction

The e-mail field extraction task was evaluated on 171 test e-mails that
represent a 20% stratified sample across all eight Woo dossiers. The
main results of the e-mail field extraction on a macro-level, averaged
over the fields FROM, TO, CC, DATE, SUBJECT and ATTACHMENT, are
presented in Table 3. In addition, the calculated ANLS is presented to
show the F1 score with more relaxed parameters.

**Table 3**

*Macro-level scores for e-mail field extraction*

| **Approach**       | **Exact macro F1** | **ANLS**  |
|--------------------|:------------------:|:---------:|
| Regex              |     **0.881**      | **0.886** |
| RobBERT (token, unweighted) |       0.312        |   0.330   |
| RobBERT (token, class-weighted) |       0.465        |   0.858   |
| GPT-5.5 (text)     |       0.741        |   0.832   |
| GPT-5.5 (vision+text)   |       0.751        |   0.841   |

The results show that for exact F1 scores, the regex baseline performs
best (0.881), followed by GPT-5.5 (vision+text) (0.751) and RobBERT (token, unweighted)
(0.312). Using the ANLS to identify approximate predictions show a small
increase for the regex baseline (0.6% increase) compared to the
fine-tuned RobBERT (token, class-weighted) model (84.5% increase) and the GPT-5.5 text
model (12.3% increase). The GPT-5.5 vision+text model performs slightly
better than the GPT-5.5 text model, though still lower than the regex
baseline.

In addition to the macro-level F1 score, looking at each field adds
nuance to the analysis to examine the strengths of each approach. The
regex baseline remains fairly consistent with a range from 0.625 at the
lower end for ATTACHMENTS and 0.991 for FROM on the higher end. GPT-5.5
text shows equally high scores on the higher bound, such as 0.983 for
DATE, though it performs poorly on ATTACHMENT at 0.264. The GPT-5.5
vision in general performs slightly worse than the text-only model with
the exception of ATTACHMENT and CC, which are both multi-fields values.
This indicates that the vision model performs better in more ambiguous
e-mail fields. The largest range in scores is recorded in the RobBERT
pipeline, where three fields scored an F1 score of 0.000, namely CC,
SUBJECT and ATTACHMENT. These scores indicate that the BERT model
performs reasonably well for some fields but breaks completely for other
fields as discussed in chapter 3.3.1 on data imbalance. With the class
weight tuning, the scores for the RobBERT model improve on all e-mail
fields without any collapsed fields. Though ATTACHMENT still shows no
correct exact predictions, the ANLS score shows that it actually finds
the most correct attachments, though still imprecise. When looking at
the ANLS scores, GPT-5.5 and the regex baseline perform best overall,
with both predicting most accurately for three e-mail fields each.

**Table 4**

*Per Field scores for e-mail field extraction*

<table>
<colgroup>
<col style="width: 19%" />
<col style="width: 8%" />
<col style="width: 8%" />
<col style="width: 8%" />
<col style="width: 7%" />
<col style="width: 0%" />
<col style="width: 8%" />
<col style="width: 8%" />
<col style="width: 6%" />
<col style="width: 8%" />
<col style="width: 8%" />
<col style="width: 8%" />
</colgroup>
<thead>
<tr>
<th></th>
<th colspan="2" style="text-align: center;"><strong>Regex</strong></th>
<th colspan="2"
style="text-align: center;"><strong>RobBERT</strong></th>
<th colspan="3" style="text-align: center;"><strong>RobBERT
(weighted)</strong></th>
<th colspan="2" style="text-align: center;"><strong>GPT-5.5
(text)</strong></th>
<th colspan="2" style="text-align: center;"><strong>GPT-5.5
(vision)</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>E-mail field</strong></td>
<td style="text-align: center;"><strong>F1</strong></td>
<td style="text-align: center;"><strong>ANLS</strong></td>
<td style="text-align: center;"><strong>F1</strong></td>
<td colspan="2" style="text-align: center;"><strong>ANLS</strong></td>
<td style="text-align: center;"><strong>F1</strong></td>
<td style="text-align: center;"><strong>ANLS</strong></td>
<td style="text-align: center;"><strong>F1</strong></td>
<td style="text-align: center;"><strong>ANLS</strong></td>
<td style="text-align: center;"><strong>F1</strong></td>
<td style="text-align: center;"><strong>ANLS</strong></td>
</tr>
<tr>
<td>FROM</td>
<td style="text-align: center;"><strong>0.991</strong></td>
<td style="text-align: center;"><strong>0.994</strong></td>
<td style="text-align: center;">0.616</td>
<td colspan="2" style="text-align: center;">0.660</td>
<td style="text-align: center;">0.617</td>
<td style="text-align: center;">0.905</td>
<td style="text-align: center;">0.988</td>
<td style="text-align: center;">0.994</td>
<td style="text-align: center;">0.976</td>
<td style="text-align: center;">0.976</td>
</tr>
<tr>
<td>TO</td>
<td style="text-align: center;">0.908</td>
<td style="text-align: center;">0.942</td>
<td style="text-align: center;">0.489</td>
<td colspan="2" style="text-align: center;">0.390</td>
<td style="text-align: center;">0.539</td>
<td style="text-align: center;">0.845</td>
<td style="text-align: center;"><strong>0.939</strong></td>
<td style="text-align: center;"><strong>0.955</strong></td>
<td style="text-align: center;">0.923</td>
<td style="text-align: center;">0.940</td>
</tr>
<tr>
<td>CC</td>
<td style="text-align: center;"><strong>0.876</strong></td>
<td style="text-align: center;"><strong>0.890</strong></td>
<td style="text-align: center;">0.000</td>
<td colspan="2" style="text-align: center;">0.000</td>
<td style="text-align: center;">0.548</td>
<td style="text-align: center;">0.864</td>
<td style="text-align: center;">0.346</td>
<td style="text-align: center;">0.603</td>
<td style="text-align: center;">0.381</td>
<td style="text-align: center;">0.603</td>
</tr>
<tr>
<td>DATE</td>
<td style="text-align: center;">0.977</td>
<td style="text-align: center;">0.983</td>
<td style="text-align: center;">0.769</td>
<td colspan="2" style="text-align: center;">0.928</td>
<td style="text-align: center;">0.778</td>
<td style="text-align: center;">0.983</td>
<td style="text-align: center;"><strong>0.983</strong></td>
<td style="text-align: center;"><strong>0.994</strong></td>
<td style="text-align: center;">0.971</td>
<td style="text-align: center;">0.971</td>
</tr>
<tr>
<td>SUBJECT</td>
<td style="text-align: center;">0.909</td>
<td style="text-align: center;">0.954</td>
<td style="text-align: center;">0.000</td>
<td colspan="2" style="text-align: center;">0.000</td>
<td style="text-align: center;">0.306</td>
<td style="text-align: center;">0.886</td>
<td style="text-align: center;"><strong>0.925</strong></td>
<td style="text-align: center;"><strong>0.988</strong></td>
<td style="text-align: center;">0.906</td>
<td style="text-align: center;">0.964</td>
</tr>
<tr>
<td>ATTACHMENT</td>
<td style="text-align: center;"><strong>0.625</strong></td>
<td style="text-align: center;">0.557</td>
<td style="text-align: center;">0.000</td>
<td colspan="2" style="text-align: center;">0.000</td>
<td style="text-align: center;">0.000</td>
<td style="text-align: center;"><strong>0.662</strong></td>
<td style="text-align: center;">0.264</td>
<td style="text-align: center;">0.461</td>
<td style="text-align: center;">0.353</td>
<td style="text-align: center;">0.594</td>
</tr>
</tbody>
</table>

Out of all fields, ATTACHMENT was the field that got predicted least
often. Worth noting is the strong increase in F1 score when given some
more leniency in both the RobBERT and GPT-5.5 approach. This increase is
not seen for the regex baseline that shows a smaller relative increase.

# Discussion

The first surprising finding of this experiment is the strong
performance of the regex baseline for e-mail segmentation, outperforming
a fine-tuned RobBERT model, and almost reaching the performance of a
state-of-the-art GPT-5.5 model. To extract e-mail fields, the regex
baseline performs even stronger than the GPT-5.5 model and considerably
stronger than the RobBERT model. While the regex baseline needs to be
contextualized with the pre-annotation scheme, the ANLS metric shows
that even when given more leniency to allow for variations, the regex
baseline performs best. The highly structured nature of the e-mail
header seems to benefit the strict pattern matching of regex functions
that does not need to learn connections between signals. In addition,
while the field values show heavy redaction the field name itself, which
is the sole trigger for the regex function, is never redacted and does
not seem to suffer from excessive OCR noise. This strictly templatic
structure of e-mail headers do not improve a model that expects semantic
structure as a training signal. Furthermore, the error analysis shows
that a lack of structural rules inhibits the token classification model
from effectively parsing e-mails. The architecture of BERT models lacks
the understanding of e-mail header structures which the training with
the dataset used in this study cannot introduce. Apart from the
architectural contribution, this finding presents a crucial implication
for reconstructing e-mail fields under Woo-requests: The structure of
e-mails can be reconstructed using simple techniques and only marginal
improvements can be expected from using deep learning models.

The same observations can be reported for the e-mail field extraction
task, where the regex pipeline outperforms all the other methods in
macro-level scores and achieving exceptionally high scores with an ANLS
of 0.886. The scores per e-mail field show that regex performs robust
over all fields, while the other approaches show a much larger range
across fields. This confirms the thesis that in order to reconstruct
e-mail fields, the highly structured and homogenous nature of e-mail
headers provide clear benefits to strict rule-based approaches such as
regex. Compounding this effect for the e-mail extraction are the heavy
redactions, which create significant OCR noise and hinder the tokenizer
in the deep learning models from matching the (sub)words to tokens it
has learnt. This leaves these tokens with weak embeddings that disrupt
the attention of the model. While weighting addressed some of the
imbalance in the data, it does not circumvent the lack of embeddings and
low token recognition. GPT-5.5 seems to suffer less from this which can
likely be contributed to the much larger training data compared to the
RobBERT model.

The second interesting finding is the drop in performance when the
GPT-5.5 pipeline was given visual context to enhance its e-mail
segmentation. While at first this finding seems to contradict the
finding of Terentieva et al. (2026) that VLM-based pipelines perform
better on PSS tasks than regex-based pipelines, the difference can be
explained by the homogeneity of documents in this study and the
compression of images to fit API call requirements. In PSS, documents
have visually distinct features that indicate a new document, whereas
e-mails share a stronger homogenous structure. Rendering the images at a
lower DPI than is optimal introduces new errors, such as hallucinations
upon partial visual cues and decreased data quality. Therefore, the
current VLM capabilities are not suitable for the proposed tasks with
the current restrictions as an end-to-end, zero-shot pipeline.

An underlying issue that not only hindered the performance of the
models’ capabilities but also hinder downstream analysis task is the OCR
noise found in almost all dossiers. The findings in this thesis are
therefore consistent with previous research that showed the effect of
NLP tasks on noisy data and bad OCR quality (Jiang et al., 2022; Van
Strien et al., 2020). While two dossiers required OCR as they did not
show any machine-readable text, the other dossiers showed strong signs
of OCR application prior to release. This suggest that PDFs have been
rasterized to increase redaction security, as Bland et al. (2022)
describes. However, the drawbacks of this practice appears only when
users of Woo dossiers attempt downstream analysis. While segmenting the
e-mails and extracting the field values is possible with a simple regex
function, the issue remains that most of the field values are heavily
distorted through the OCR noise. E-mail addresses that showed as
“.,..,@--\\-\\-m=in..:.;:d=e"\\-".f. : ....:.:.ln\>” were not an anomaly
but present a structural issue of the redaction workflow for FOIA
documents. This makes analysis using transformer-based models more
challenging as many of the tokens lose significance through the noise.
While simple regex functions prove beneficial to reconstruct e-mail
metadata, the usability of the metadata decreases significantly through
the practices of redaction.

The lack of usability of FOIA-requested documents that was identified by
other scholars (e.g. Luscombe & Walby, 2017; Mario & Kilty, 2025; Spáč
et al., 2025) can also be argued in the case of Dutch Woo e-mail
releases. The OCR noise leaves the data in formats that are difficult to
use for regular citizens and data scientists alike and do not contribute
to effective transparency. Whether intentional or not, the production
workflow seems to disregard the needs of end-users of such requests.

The findings of this thesis lead to a fundamental conclusion for e-mail
archives based on FOIA-requested e-mails: effective transparency can be
greatly improved if annotation practices follow a structured,
machine-readable and token-aware process. An illustrative example
emerged during the analysis per dossiers, where one dossier showed
virtually no noise that could be attributed to OCR and a near perfect
recall score across all approaches. Visual inspection indicated the
annotation practices that lead to this result.

Firstly, the embedded text seemed to be a native text layer and not
rasterized and re-constructed using OCR. This prevented any OCR noise
from occurring and which could decrease transformer performance. The
redaction bars were marked with ‘5.1.2’ redaction codes which were
machine readable as well and are placed in-line with the other text. A
consistent redaction labelling scheme would also help to make redactions
more linguistically coherent which would aid transformer-based models.
This contrasts other practices in dossiers where redaction bars are
drawn as free rectangles and can therefore corrupt more than one line at
a time. Lastly, all e-mail chains started as a new page, effectively
eliminating the need for a separate e-mail segmentation task.

The example of this dossier exemplifies the broader tension that this
thesis highlights: With a more robust and structured FOIA document
production and publication workflow, the need for computationally
expensive models, or laborious training of models is essentially
eliminated. The technical contribution of this thesis should be rendered
obsolete as they try to re-establish structure that once existed. The
Clinton e-mail release discussed earlier shows that when e-mails are
released with metadata and machine-readable text, users can create their
own products for analyses and extract valuable insights with
computational methods. The format of release is a policy decision that
can greatly advance the transparency goals set by governments around the
world.

# Conclusion

This thesis set out to explore whether machine learning methods can
reconstruct e-mails from Dutch Woo dossiers into structured, navigable
records. The second exploration set out to recommend structural
improvements in the release of such documents to facilitate extraction.

The first question can be confirmed as the best performing model,
GPT-5.5 with textual input, scores an exact F1 score of 0.971 on e-mail
segmentation. E-mail field extraction is performed best by the GPT-5.5
model with text and visual input with an exact F1 score of 0.741, while
the trained RobBERT model with class weight scored an ANLS F1 score of
0.914. These results are promising to consider the structural challenges
of Woo dossiers such as OCR noise and redaction bars. However, the
research question set out to consider feasibility and usability for
Woo-stakeholders as well, under which lens the regex function performs
exceptionally strong and clearly outperforms the other models in
computing requirements and cost. The e-mail segmentation was conducted
with an F1 of 0.924 and scored highest of all approaches on e-mail field
extraction, with an exact F1 of 0.881 and an ANLS F1 of 0.963. This
inverts the assumption that increasingly capable models will solve the
usability problem faced by FOIA requested documents.

The second question is more difficult to answer, as the government
practices of e-mail redaction and Woo disclosure are rather opaque.
Still, the analysis of the most common failure modes of the three
approaches highlights the necessity of three structural improvements in
the Woo disclosure process. First is the need for a native pdf text
layer that was not re-established with OCR. Second is the redaction
workflow, which ideally follows a structured in-line approach that
preserves the native text layer and layout structure. Lastly, e-mails
should be concatenated at the page level rather than line level, which
would eliminate the need for edge case e-mail segmentation.

The thesis makes three contributions to the literature of FOIA requested
documents as living archives. First, it releases the first publicly
available dataset of Dutch Woo e-mails with spans and annotated field
values. While small in size compared to other datasets on similar tasks,
it functions as a starting point to test various e-mail metadata
reconstruction methods and introduces more difficult cases such as
redaction bars and OCR text layers that are fundamental to many
FOIA-related disclosures. Secondly, it makes a methodological
contribution by comparing three approaches to the task of e-mail
segmentation and e-mail field extraction and makes an important
contribution in the quest of architecture-task fit in a time where
larger generalist models are becoming predominant. Lastly, the
discussion of the results presents a diagnosis on the failure modes of
effective e-mail metadata extraction. It critiques the detrimental
effect of the process of disclosure that occurs within government
agencies and provides remedies to improve disclosures for downstream
analysis tasks. It contributes to the transparency-by-design ethos that
it is not technical superiority, but the nature of the design itself
that contributes to transparency (Janssen et al., 2017).

While the Woo creates the legal framework of disclosure and thus creates
a ground for transparency, it does not ensure the processes that would
satisfy the usability for stakeholders and end-users. Downstream of the
legal grounds lies an organizational reality, where civil servants
working to comply with the law and minimize liability by ensuring that
redactions are secure have resulted in a workflow that make Woo dossiers
difficult to use for its end-users. It is in this organizational reality
where the smallest and most effective improvements can be made that will
encourage transparency as an outcome of its intrinsic design.

# References

Allegrezza, S. (2022). *Recent Developments on E–Mail Preservation:
Towards the Ultimate Solution?*
https://cris.unibo.it/handle/11585/907520

Altena, B. (2020, February 21). *Lobbyhoogleraar: Contact Schiphol en
minister “te close.”* EenVandaag.
https://eenvandaag.avrotros.nl/artikelen/lobbyhoogleraar-contact-schiphol-en-minister-te-close-122268

Ates, Ö. (2025). *Vision-Language Models for OCR Beyond Plain Text*
\[Bachelor Thesis\]. University of Amsterdam.

Banikarim, S. (2026, February 13). Laurels and Darts: You’ve got Jmail.
*Columbia Journalism Review*.
https://www.cjr.org/laurels-and-darts/youve-got-jmail-tool-clone-epstein-files-parse-mississippi-today-free-press-florida-medical-waiting-list-home-health-care.php

Baron, J. R. (2025). Using AI in providing greater access to the U.S.
government’s email: A progress report. *AI & SOCIETY*, *40*(7),
5359–5372. https://doi.org/10.1007/s00146-025-02256-3

Berliner, D. D., Ingrams, A., & Piotrowski, S. J. (2019). The Future of
FOIA in an Open Government World: Implications of the Open Government
Agenda for Freedom of Information Policy and Implementation. *Villanova
Law Review*, *63*(5).
https://www.villanovalawreview.com/article/10569-the-future-of-foia-in-an-open-government-world-implications-of-the-open-government-agenda-for-freedom-of-information-policy-and-implementation

Bland, M., Iyer, A., & Levchenko, K. (2022). Story Beyond the Eye: Glyph
Positions Break PDF Text Redaction. *Proceedings on Privacy Enhancing
Technologies*, *2023*(3), 43–61.
https://doi.org/10.56553/popets-2023-0069

Cucciniello, M., & Nasi, G. (2014). Transparency for Trust in
Government: How Effective is Formal Transparency? *International Journal
of Public Administration*, *37*(13), 911–921.
https://doi.org/10.1080/01900692.2014.949754

De Felice, R., & Garretson, G. (2018). Politeness at Work in the Clinton
Email Corpus: A First Look at the Effects of Status and Gender. *Corpus
Pragmatics*, *2*(3), 221–242. https://doi.org/10.1007/s41701-018-0034-2

de Vries, W., van Cranenburgh, A., Bisazza, A., Caselli, T., van Noord,
G., & Nissim, M. (2019, December 19). *BERTje: A Dutch BERT Model*.
arXiv.Org. https://arxiv.org/abs/1912.09582v1

Delobelle, P., & Remy, F. (2024). RobBERT-2023: Keeping Dutch Language
Models Up-To-Date at a Lower Cost Thanks to Model Conversion.
*Computational Linguistics in the Netherlands Journal*, *13*, 193–203.

Delobelle, P., Winters, T., & Berendt, B. (2020). RobBERT: A Dutch
RoBERTa-based Language Model. In T. Cohn, Y. He, & Y. Liu (Eds.),
*Findings of the Association for Computational Linguistics: EMNLP 2020*
(pp. 3255–3265). Association for Computational Linguistics.
https://doi.org/10.18653/v1/2020.findings-emnlp.292

DTAI KU Leuven. (2023, December 5).
*DTAI-KULeuven/robbert-2023-dutch-base · Hugging Face*.
https://huggingface.co/DTAI-KULeuven/robbert-2023-dutch-base

Gardazi, N. M., Daud, A., Malik, M. K., Bukhari, A., Alsahfi, T., &
Alshemaimri, B. (2025). BERT applications in natural language
processing: A review. *Artificial Intelligence Review*, *58*(6), 166.
https://doi.org/10.1007/s10462-025-11162-5

Guruprasad, P., Chowdhury, S., Sikka, H., Sharma, M., Lu, H., Rivera,
S., Khurana, A., Ren, H., & Wang, Y. (2025). *Benchmarking the
Generality of Vision-Language-Action Models* (arXiv:2512.11315). arXiv.
https://doi.org/10.48550/arXiv.2512.11315

Hannah, M. N. (2026, May 8). *Conspiracy theorists are building AI
interfaces to the Epstein files – and presenting their views as data
analysis*. The Conversation. https://doi.org/10.64628/AAI.3cmcpjpnw

Hartholt, S. (2025, December 12). *Plan om Woo-verzoeken in te perken
oogst felle kritiek*. Binnenlands Bestuur.
https://www.binnenlandsbestuur.nl/digitaal/plan-om-woo-in-te-perken-oogst-kritiek

Hendrickx, F., & Kreling, T. (2022, March 23). *Hoe Hugo de Jonge zich
actief bemoeide met de mondkapjesdeal van Sywert van Lienden*. De
Volkskrant. Volkskrant.
https://www.volkskrant.nl/kijkverder/v/2022/hoe-hugo-de-jonge-zich-actief-bemoeide-met-de-mondkapjesdeal-van-sywert-van-lienden~v497075/

Honee, L. F. D., & Drahmann, A. (2024). The right to access public
information: A legal comparison between Sweden and the Netherlands.
*Review of European Administrative Law*, *2024*(3/4), 7–38.
https://doi.org/10.7590/187479824X17326230137627

Jaillant, L. (2022). How can we make born-digital and digitised archives
more accessible? Identifying obstacles and solutions. *Archival
Science*, *22*(3), 417–436. https://doi.org/10.1007/s10502-022-09390-7

Janssen, M., Matheus, R., Longo, J., & Weerakkody, V. (2017).
Transparency-by-design as a foundation for open government.
*Transforming Government: People, Process and Policy (Online)*, *11*(1),
2–8. https://doi.org/10.1108/TG-02-2017-0015

Jeon, H., Kim, K., & Shin, J. (2025, October 8). *RedacBench: Can AI
Erase Your Secrets?* The Fourteenth International Conference on Learning
Representations. https://openreview.net/forum?id=wf73W2xatC

Jiang, M., D’Souza, J., Auer, S., & Downie, J. S. (2022). Evaluating
BERT-based Scientific Relation Classifiers for Scholarly Knowledge Graph
Construction on Digital Library Collections. *International Journal on
Digital Libraries*, *23*(2), 197–215.
https://doi.org/10.1007/s00799-021-00313-y

Jin, A. Y. (2025). *Github: Yonsei-sslab/EmailBERT* \[Python\]. Secure
Systems Lab @ Yonsei University.
https://github.com/yonsei-sslab/EmailBERT

Johnston, J. A., Wallace, D. A., & Punzalan, R. L. (2019). Messages
sent, and received? Changing perspectives and policies on US federal
email as record and the limits of archival accountability. *Archival
Science*, *19*(4), 309–329. https://doi.org/10.1007/s10502-019-09318-8

Jurafsky, D., & Martin, J. H. (2026). *Speech and Language Processing*
(3rd ed.). Stanford. https://web.stanford.edu/~jurafsky/slp3/

Keraghel, I., Morbieu, S., & Nadif, M. (2024). *Recent Advances in Named
Entity Recognition: A Comprehensive Survey and Comparative Study*
(arXiv:2401.10825). arXiv. https://doi.org/10.48550/arXiv.2401.10825

Leopold, J. (2026, February 6). Epstein Files Review Was Totally
Chaotic. *Bloomberg.Com*.
https://www.bloomberg.com/news/newsletters/2026-02-06/epstein-files-review-was-chaotic

Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2018). *Focal
Loss for Dense Object Detection* (arXiv:1708.02002). arXiv.
https://doi.org/10.48550/arXiv.1708.02002

Lingren, T., Deleger, L., Molnar, K., Zhai, H., Meinzen-Derr, J.,
Kaiser, M., Stoutenborough, L., Li, Q., & Solti, I. (2014). Evaluating
the impact of pre-annotation on annotation speed and potential bias:
Natural language processing gold standard development for clinical named
entity recognition in clinical trial announcements. *Journal of the
American Medical Informatics Association: JAMIA*, *21*(3), 406–413.
https://doi.org/10.1136/amiajnl-2013-001837

Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O.,
Lewis, M., Zettlemoyer, L., & Stoyanov, V. (2019). *RoBERTa: A Robustly
Optimized BERT Pretraining Approach* (arXiv:1907.11692). arXiv.
https://doi.org/10.48550/arXiv.1907.11692

Luscombe, A., & Walby, K. (2017). Theorizing freedom of information: The
live archive, obfuscation, and actor-network theory. *Government
Information Quarterly*, *34*(3), 379–387.
https://doi.org/10.1016/j.giq.2017.09.003

Mahlawi, A. Q., & Sasi, S. (2017). Structured data extraction from
emails. *2017 International Conference on Networks & Advances in
Computational Technologies (NetACT)*, 323–328.
https://doi.org/10.1109/NETACT.2017.8076789

Mario, B., & Kilty, J. (2025). A Right to Know? Using Access to
Information as Method in Critical Criminological Research. *Qualitative
Inquiry*, *31*(5), 492–504. https://doi.org/10.1177/10778004241256140

Mikulová, M., Straka, M., Štěpánek, J., Štěpánková, B., & Hajič, J.
(2023). *Quality and Efficiency of Manual Annotation: Pre-annotation
Bias* (arXiv:2306.09307). arXiv.
https://doi.org/10.48550/arXiv.2306.09307

Millhiser, I. (2019, October 22). *The embarrassing epilogue to the
media’s obsession with Hillary Clinton’s emails*. Vox.
https://www.vox.com/policy-and-politics/2019/10/22/20924795/hillary-clinton-emails-new-york-times-state-department

Ministerie van Binnenlandse Zaken en Koninkrijksrelaties. (2024).
*Woo-instructie voo het behandelen van Woo-verzoeken*. Ministerie van
Binnenlandse Zaken en Koninkrijksrelaties.

Moon, M. J. (2020). Shifting from Old Open Government to New Open
Government: Four Critical Dimensions and Case Illustrations. *Public
Performance & Management Review*, *43*(3), 535–559.
https://doi.org/10.1080/15309576.2019.1691024

Nemoto, S., Kitada, S., & Iyatomi, H. (2025). Majority or Minority: Data
Imbalance Learning Method for Named Entity Recognition. *IEEE Access*,
*13*, 9902–9909. https://doi.org/10.1109/ACCESS.2024.3522972

Noveck, B. S. (2016, November 21). *Is Open Data the Death of FOIA?*
Yale Law Journal.
https://yalelawjournal.org/essay/is-open-data-the-death-of-foia

OpenVWS. (2026, May 19). *Alle COVID-19 gerelateerde besluiten \|
OpenVWS*. Ministerie van Volksgezondheid, Welzijn en Sport.
https://open.minvws.nl/thema/covid-19

Ouyang, L., Qu, Y., Zhou, H., Zhu, J., Zhang, R., Lin, Q., Wang, B.,
Zhao, Z., Jiang, M., Zhao, X., Shi, J., Wu, F., Chu, P., Liu, M., Li,
Z., Xu, C., Zhang, B., Shi, B., Tu, Z., & He, C. (2025). *OmniDocBench:
Benchmarking Diverse PDF Document Parsing with Comprehensive
Annotations* (arXiv:2412.07626). arXiv.
https://doi.org/10.48550/arXiv.2412.07626

Pak, I., & Teh, P. L. (2018). Text Segmentation Techniques: A Critical
Review. In I. Zelinka, P. Vasant, V. H. Duy, & T. T. Dao (Eds.),
*Innovative Computing, Optimization and Its Applications: Modelling and
Simulations* (pp. 167–181). Springer International Publishing.
https://doi.org/10.1007/978-3-319-66984-7_10

Peer, D., Schöpf, P., Nebendahl, V., Rietzler, A., & Stabinger, S.
(2025). *ANLS\*—A Universal Document Processing Metric for Generative
Large Language Models* (arXiv:2402.03848). arXiv.
https://doi.org/10.48550/arXiv.2402.03848

Qiu, X., Sun, T., Xu, Y., Shao, Y., Dai, N., & Huang, X. (2020).
Pre-trained models for natural language processing: A survey. *Science
China Technological Sciences*, *63*(10), 1872–1897.
https://doi.org/10.1007/s11431-020-1647-3

Ruijer, E., Détienne, F., Baker, M., Groff, J., & Meijer, A. J. (2020).
The Politics of Open Government Data: Understanding Organizational
Responses to Pressure for More Transparency. *The American Review of
Public Administration*, *50*(3), 260–274.
https://doi.org/10.1177/0275074019888065

Ruijer, E., Grimmelikhuijsen, S., van den Berg, J., & Meijer, A. (2020).
Open data work: Understanding open data usage from a practice lens.
*International Review of Administrative Sciences*, *86*(1), 3–19.
https://doi.org/10.1177/0020852317753068

Rust, P., Pfeiffer, J., Vulić, I., Ruder, S., & Gurevych, I. (2020,
December 31). *How Good is Your Tokenizer? On the Monolingual
Performance of Multilingual Language Models*. arXiv.Org.
https://arxiv.org/abs/2012.15613v2

Schohaus, B. (2026, January 18). *Wat de tegenstanders van de Woo liever
niet vertellen*. Follow the Money - Platform voor
onderzoeksjournalistiek.
https://www.ftm.nl/artikelen/wat-de-tegenstanders-van-de-woo-liever-niet-vertellen

Sebők, M., Kovács, V., Bánóczy, M., Eriksen, D. M., Neptune, N., &
Roussille, P. (2025). *Beyond Token Limits: Assessing Language Model
Performance on Long Text Classification* (arXiv:2509.10199). arXiv.
https://doi.org/10.48550/arXiv.2509.10199

Sigdyal, P. (2016, March 27). *What data reveals about Hillary Clinton’s
emails*. CNBC.
https://www.cnbc.com/2016/03/27/hillary-clintons-emails-what-does-the-data-show.html

Spáč, P., Pastarmadzhieva, D., & Zagrapan, J. (2025). Freedom of
information and the volume of requested data: An experimental study.
*Government Information Quarterly*, *42*(2), 102030.
https://doi.org/10.1016/j.giq.2025.102030

Strop, J.-H. (2020, December 24). *Toeslagenaffaire: Commissie-Donner
trok conclusies in strijd met eigen onderzoek*. Follow the Money -
Platform voor onderzoeksjournalistiek.
https://www.ftm.nl/artikelen/toeslagenaffaire-wob-commissie-donner

Terentieva, Y., Wechsler, J., de Vries, C., Jans, T., & Kamps, J.
(2026). *From Formal Transparency to Practical Interpretability: WOOLens
for Open Government Data*. 9.

U.S. Department of Justice. (2026, January 30). *Office of Public
Affairs \| Department of Justice Publishes 3.5 Million Responsive Pages
in Compliance with the Epstein Files Transparency Act \| United States
Department of Justice*.
https://www.justice.gov/opa/pr/department-justice-publishes-35-million-responsive-pages-compliance-epstein-files

Vajjala, S., & Shimangaud, S. (2025). *Text Classification in the LLM
Era—Where do we stand?* (arXiv:2502.11830). arXiv.
https://doi.org/10.48550/arXiv.2502.11830

van Heusden, R., de Ruijter, A., Majoor, R., & Marx, M. (2023).
Detection of Redacted Text in Legal Documents. In O. Alonso, H. Cousijn,
G. Silvello, M. Marrero, C. Teixeira Lopes, & S. Marchesin (Eds.),
*Linking Theory and Practice of Digital Libraries* (pp. 310–316).
Springer Nature Switzerland.
https://doi.org/10.1007/978-3-031-43849-3_28

Van Heusden, R., Kamps, J., & Marx, M. (2024). OpenPSS: An Open Page
Stream Segmentation Benchmark. In A. Antonacopoulos, A. Hinze, B.
Piwowarski, M. Coustaty, G. M. Di Nunzio, F. Gelati, & N. Vanderschantz
(Eds.), *Linking Theory and Practice of Digital Libraries* (Vol. 15177,
pp. 413–429). Springer Nature Switzerland.
https://doi.org/10.1007/978-3-031-72437-4_24

van Heusden, R., Meijer, K., & Marx, M. (2025a). Redacted text detection
using neural image segmentation methods. *International Journal on
Document Analysis and Recognition*, *28*(4), 597–607.
https://doi.org/10.1007/s10032-025-00513-1

van Heusden, R., Meijer, K., & Marx, M. (2025b). Redacted text detection
using neural image segmentation methods. *International Journal on
Document Analysis and Recognition*, *28*(4), 597–607.
https://doi.org/10.1007/s10032-025-00513-1

van Leeuwen, M., Haak, K., Saygili, G., Postma, E., & Ong, S. (2025). A
Note On The Stability Of The Focal Loss. *Transactions on Machine
Learning Research*.

Van Strien, D., Beelen, K., Ardanuy, M., Hosseini, K., McGillivray, B.,
& Colavizza, G. (2020). Assessing the Impact of OCR Quality on
Downstream NLP Tasks: *Proceedings of the 12th International Conference
on Agents and Artificial Intelligence*, 484–496.
https://doi.org/10.5220/0009169004840496

WikiLeaks. (2026, October 5). *Hillary Clinton Email Archive*.
WikiLeaks.
https://wikileaks.org/clinton-emails/?q=classified&mfrom=&mto=&title=&notitle=&date_from=&date_to=&nofrom=&noto=&count=50&sort=0#searchresult

Zhang, J., Huang, Y., Liu, S., Gao, Y., & Hu, X. (2025). *Do BERT-Like
Bidirectional Models Still Perform Better on Text Classification in the
Era of LLMs?* (arXiv:2505.18215). arXiv.
https://doi.org/10.48550/arXiv.2505.18215

Zhang, Q., Wang, B., Huang, V. S.-J., Zhang, J., Wang, Z., Liang, H.,
He, C., & Zhang, W. (2024). *Document Parsing Unveiled: Techniques,
Challenges, and Prospects for Structured Information Extraction*
(arXiv:2410.21169). arXiv. https://doi.org/10.48550/arXiv.2410.21169
