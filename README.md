#  Tool–Tissue Interaction (TTI) Detection, Isolation, and Identification in Minimally Invasive Surgery

## Overview

This repository contains the implementation of a system for **automatic detection, isolation, and identification of tool–tissue interactions** during **laparoscopic cholecystectomy (LC)** procedures.  
The goal is to enhance computer-assisted surgical systems by accurately analyzing how surgical tools interact with tissue, supporting **surgical workflow recognition**, **skill assessment**, and **context-aware assistance**.

---

##  Objectives

- **Detection** – Detection of a TTI event type with mask, confidence and bounding box  
- **Isolation** – Matching the tool and the tissue that are interacting without the
 indication of the type of interaction  
- **Identification** – Adding to the
 previous information also the type of interaction (retract, dissect, etc...)  

## Organization
This repository is organised as follows:

in the **Common Code** folder there is the code to train, evaluate and run the inference of the fully-supervised frame-by-frame pipeline.
In the **Carlo** folder there is the code to evaluate and run in real time the supervised multi-frame model.
In the **Filippo/Test** there is the code to create the weakly supervised dataset and the one to train, run and evaluate the respective pipeline

