# Synthetic Test Dataset for EnhancedTabbycat (Speech Analytics Testing)

This dataset contains synthetic, fictional test data designed to populate your local tournament with:
- **2 Institutions**: University of Veritas (`VER`), Apex Debate Institute (`APX`)
- **4 Teams**: 2 from Veritas (`Veritas 1`, `Veritas 2`), 2 from Apex (`Apex Alpha`, `Apex Beta`)
- **8 Speakers**: 2 per team (using dummy email addresses `@example.com`)
- **2 Venues / Rooms**: `Room 101`, `Room 102`
- **2 Adjudicators**: 1 per institution

All data is strictly fictional and uses `example.com` domains.

---

## What is the "Import Data" Option in the Admin Sidebar?

In Tabbycat, navigating to **Setup > Import Data** (`/<tournament_slug>/admin/importer/simple/`) opens the **Simple Importer**.
- It is **not** an XML tournament archive importer.
- It is specifically intended for **bulk registration data entry** (Institutions, Teams, Adjudicators, Rooms/Venues) through web-based wizards.
- (Archive export/import is located separately under `/<tournament_slug>/admin/importer/export/` and `/archive/` for full database XML dumps).

---

## Method 1: Web Interface Workflow (Simple Importer - Recommended)

The easiest, standard Tabbycat UI workflow requires **no command line** and works with your existing `Verbatim 26` tournament:

### Step 1: Import Institutions
1. In the tournament sidebar, open **Setup > Import Data**.
2. Click **Add Institutions** (`/verbatim-26/admin/importer/simple/institutions/`).
3. Paste the contents of `test-data/institutions_simple_paste.txt` into the box:
   ```text
   University of Veritas,VER,North
   Apex Debate Institute,APX,South
   ```
4. Click **Next Step**, review the parsed fields, and click **Save All Institutions**.

### Step 2: Import Rooms / Venues (Required for debates)
1. In **Setup > Import Data**, click **Add Rooms**.
2. Paste:
   ```text
   Room 101,10
   Room 102,9
   ```
3. Click **Next Step**, review, and click **Save All Rooms**.

### Step 3: Import Teams & Speakers
1. In **Setup > Import Data**, click **Add Teams**.
2. In Step 1 (**Numbers**):
   - Under **University of Veritas (VER)**: enter `2`
   - Under **Apex Debate Institute (APX)**: enter `2`
   - Click **Next Step**.
3. In Step 2 (**Details**):
   - Fill in or verify team names and speaker names:
     - **Team 1 (Veritas)**:
       - Name: `Veritas 1`
       - Speakers:
         ```text
         Alex Mercer
         Jordan Lee
         ```
       - Emails (optional):
         ```text
         alex.mercer@example.com
         jordan.lee@example.com
         ```
     - **Team 2 (Veritas)**:
       - Name: `Veritas 2`
       - Speakers:
         ```text
         Taylor Brooks
         Morgan Chen
         ```
       - Emails (optional):
         ```text
         taylor.brooks@example.com
         morgan.chen@example.com
         ```
     - **Team 3 (Apex)**:
       - Name: `Apex Alpha`
       - Speakers:
         ```text
         Casey Rivera
         Samira Khan
         ```
       - Emails (optional):
         ```text
         casey.rivera@example.com
         samira.khan@example.com
         ```
     - **Team 4 (Apex)**:
       - Name: `Apex Beta`
       - Speakers:
         ```text
         Riley Harper
         Jamie Vance
         ```
       - Emails (optional):
         ```text
         riley.harper@example.com
         jamie.vance@example.com
         ```
4. Click **Save All Teams**.

### Step 4: Import Adjudicators (Optional, needed to auto-allocate panel)
1. In **Setup > Import Data**, click **Add Adjudicators**.
2. Enter `1` for Veritas, `1` for Apex.
3. In Step 2, enter:
   - Veritas: `Dana Scully`, rating `5`
   - Apex: `Fox Mulder`, rating `5`
4. Click **Save All Adjudicators**.

### Step 5: Generate 2 Debates (Round 1)
1. In the sidebar, expand **Round 1**.
2. Click **Availability** (`/verbatim-26/round/1/availability/`).
3. Click the green button **Generate Draw**.
4. Tabbycat will pair the 4 teams into **2 Debates** (e.g. `Veritas 1 vs Apex Beta` and `Apex Alpha vs Veritas 2`) and assign them to `Room 101` and `Room 102`.
5. Once generated, go to **Speech Analytics > Upload Recording**:
   - The debate dropdown will now list both debates!
   - The speaker dropdown will list the speakers competing in the selected tournament.

---

## Method 2: CLI Bulk Import (`manage.py importtournament`)

If you ever want to reset or spin up a dedicated test tournament directly from CSV files using Tabbycat's `anorak` importer CLI:

```bash
env -u DEBUG pipenv run python manage.py importtournament test-data --slug verbatim-26-synth --name "Verbatim 26 Synthetic" --auto-rounds 2 --force
```

This uses the included CSV files (`institutions.csv`, `speakers.csv`, `venues.csv`, `adjudicators.csv`, `.importer`).
