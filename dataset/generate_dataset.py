"""
generate_dataset.py
Generates the instruction-tuning dataset for Resume Intelligence (resume -> structured JSON parsing).

Design principles:
- Every example uses the SAME instruction template + output schema, so the model learns
  one consistent contract (critical for reliable JSON output after fine-tuning).
- `input` is realistic, synthetic resume text covering a specific edge case.
- `output` is the ground-truth structured extraction, including explicit `null` for
  missing fields (never invented) and empty lists `[]` where a section is absent.
- Category tags are stored separately (not in the training record) purely for our own
  dataset-quality bookkeeping / stratified evaluation splits.

Run: python generate_dataset.py
Produces: resume_parsing_dataset.jsonl
"""
import json

INSTRUCTION = (
    "You are a resume parsing engine. Extract structured information from the resume "
    "text below into a JSON object with exactly these fields: name, email, phone, "
    "location, summary, skills (list), education (list of {degree, institution, year}), "
    "experience (list of {title, company, duration, description}), certifications (list), "
    "projects (list of {name, description}), links (list). "
    "If a field is not present in the text, use null for single values or [] for lists. "
    "Never invent information that is not in the text. Respond with JSON only."
)

def rec(input_text, output, category):
    return {
        "instruction": INSTRUCTION,
        "input": input_text.strip(),
        "output": output,
        "_category": category,  # stripped before final training file; kept for QA bookkeeping
    }

examples = []

# ---------- 1-5: Standard chronological resumes ----------
examples.append(rec(
    """Ananya Rao
ananya.rao@gmail.com | +91-9876543210 | Bengaluru, India

SUMMARY
Backend engineer with 4 years of experience building scalable APIs in Python and Go.

SKILLS
Python, Go, PostgreSQL, Docker, Kubernetes, AWS

EDUCATION
B.Tech in Computer Science, VIT Vellore, 2019

EXPERIENCE
Software Engineer II, Razorpay, Jun 2021 - Present
Built and maintained payment reconciliation services handling 2M transactions/day.

Software Engineer, Freshworks, Jul 2019 - May 2021
Developed REST APIs for the CRM product used by 10k+ businesses.

CERTIFICATIONS
AWS Certified Solutions Architect - Associate""",
    {
        "name": "Ananya Rao",
        "email": "ananya.rao@gmail.com",
        "phone": "+91-9876543210",
        "location": "Bengaluru, India",
        "summary": "Backend engineer with 4 years of experience building scalable APIs in Python and Go.",
        "skills": ["Python", "Go", "PostgreSQL", "Docker", "Kubernetes", "AWS"],
        "education": [{"degree": "B.Tech in Computer Science", "institution": "VIT Vellore", "year": "2019"}],
        "experience": [
            {"title": "Software Engineer II", "company": "Razorpay", "duration": "Jun 2021 - Present",
             "description": "Built and maintained payment reconciliation services handling 2M transactions/day."},
            {"title": "Software Engineer", "company": "Freshworks", "duration": "Jul 2019 - May 2021",
             "description": "Developed REST APIs for the CRM product used by 10k+ businesses."}
        ],
        "certifications": ["AWS Certified Solutions Architect - Associate"],
        "projects": [],
        "links": []
    }, "standard_chronological"))

examples.append(rec(
    """John Miller
Email: j.miller84@outlook.com
Phone: (312) 555-0192
Chicago, IL

Experienced project manager with a decade in construction and infrastructure projects.

Skills: Primavera P6, MS Project, Budgeting, Stakeholder Management, Risk Assessment

Work History
Senior Project Manager - Turner Construction (2016 - Present)
Lead delivery of commercial construction projects worth $50M+.

Project Manager - Skanska (2012 - 2016)
Managed mid-size infrastructure builds across the Midwest.

Education
MBA, University of Chicago Booth School of Business, 2011
BS Civil Engineering, Purdue University, 2007""",
    {
        "name": "John Miller",
        "email": "j.miller84@outlook.com",
        "phone": "(312) 555-0192",
        "location": "Chicago, IL",
        "summary": "Experienced project manager with a decade in construction and infrastructure projects.",
        "skills": ["Primavera P6", "MS Project", "Budgeting", "Stakeholder Management", "Risk Assessment"],
        "education": [
            {"degree": "MBA", "institution": "University of Chicago Booth School of Business", "year": "2011"},
            {"degree": "BS Civil Engineering", "institution": "Purdue University", "year": "2007"}
        ],
        "experience": [
            {"title": "Senior Project Manager", "company": "Turner Construction", "duration": "2016 - Present",
             "description": "Lead delivery of commercial construction projects worth $50M+."},
            {"title": "Project Manager", "company": "Skanska", "duration": "2012 - 2016",
             "description": "Managed mid-size infrastructure builds across the Midwest."}
        ],
        "certifications": [],
        "projects": [],
        "links": []
    }, "standard_chronological"))

examples.append(rec(
    """PRIYA MENON
priya.menon@yahoo.co.in | 9845012345 | Kochi, Kerala

Digital marketing specialist focused on SEO and performance campaigns.

CORE SKILLS
SEO, Google Ads, Meta Ads Manager, HubSpot, Content Strategy, Google Analytics

PROFESSIONAL EXPERIENCE
Marketing Manager | Byju's | Mar 2020 - Present
Scaled organic traffic 3x through content and technical SEO overhaul.

SEO Executive | Zoho | Jan 2018 - Feb 2020
Managed keyword strategy for 5 product lines.

EDUCATION
MBA Marketing, Christ University, Bangalore, 2017""",
    {
        "name": "Priya Menon",
        "email": "priya.menon@yahoo.co.in",
        "phone": "9845012345",
        "location": "Kochi, Kerala",
        "summary": "Digital marketing specialist focused on SEO and performance campaigns.",
        "skills": ["SEO", "Google Ads", "Meta Ads Manager", "HubSpot", "Content Strategy", "Google Analytics"],
        "education": [{"degree": "MBA Marketing", "institution": "Christ University, Bangalore", "year": "2017"}],
        "experience": [
            {"title": "Marketing Manager", "company": "Byju's", "duration": "Mar 2020 - Present",
             "description": "Scaled organic traffic 3x through content and technical SEO overhaul."},
            {"title": "SEO Executive", "company": "Zoho", "duration": "Jan 2018 - Feb 2020",
             "description": "Managed keyword strategy for 5 product lines."}
        ],
        "certifications": [],
        "projects": [],
        "links": []
    }, "standard_chronological"))

examples.append(rec(
    """David Chen
david.chen.dev@gmail.com | 415-555-0148 | San Francisco, CA
linkedin.com/in/davidchen | github.com/dchen

SUMMARY
Full-stack developer specializing in React and Node.js, 3 years experience.

SKILLS
JavaScript, TypeScript, React, Node.js, MongoDB, GraphQL

EXPERIENCE
Full Stack Developer, Stripe, Aug 2022 - Present
Built internal tooling dashboards used by 200+ support agents.

Junior Developer, Plaid, Jun 2021 - Jul 2022
Contributed to the developer-facing API documentation portal.

EDUCATION
BS Computer Science, UC Berkeley, 2021""",
    {
        "name": "David Chen",
        "email": "david.chen.dev@gmail.com",
        "phone": "415-555-0148",
        "location": "San Francisco, CA",
        "summary": "Full-stack developer specializing in React and Node.js, 3 years experience.",
        "skills": ["JavaScript", "TypeScript", "React", "Node.js", "MongoDB", "GraphQL"],
        "education": [{"degree": "BS Computer Science", "institution": "UC Berkeley", "year": "2021"}],
        "experience": [
            {"title": "Full Stack Developer", "company": "Stripe", "duration": "Aug 2022 - Present",
             "description": "Built internal tooling dashboards used by 200+ support agents."},
            {"title": "Junior Developer", "company": "Plaid", "duration": "Jun 2021 - Jul 2022",
             "description": "Contributed to the developer-facing API documentation portal."}
        ],
        "certifications": [],
        "projects": [],
        "links": ["linkedin.com/in/davidchen", "github.com/dchen"]
    }, "standard_chronological"))

examples.append(rec(
    """Fatima Al-Sayed
fatima.alsayed@hotmail.com | +971-50-1234567 | Dubai, UAE

Human Resources professional with 6 years in talent acquisition across the GCC region.

Skills: Recruitment, Onboarding, HRIS (Workday), Employee Relations, Arabic/English Bilingual

Experience
HR Business Partner, Emirates NBD, 2019 - Present
Lead full-cycle recruitment for retail banking division, 150+ hires annually.

Talent Acquisition Specialist, du Telecom, 2017 - 2019
Sourced technical and non-technical candidates across UAE.

Education
Bachelor of Business Administration, American University of Sharjah, 2016""",
    {
        "name": "Fatima Al-Sayed",
        "email": "fatima.alsayed@hotmail.com",
        "phone": "+971-50-1234567",
        "location": "Dubai, UAE",
        "summary": "Human Resources professional with 6 years in talent acquisition across the GCC region.",
        "skills": ["Recruitment", "Onboarding", "HRIS (Workday)", "Employee Relations", "Arabic/English Bilingual"],
        "education": [{"degree": "Bachelor of Business Administration", "institution": "American University of Sharjah", "year": "2016"}],
        "experience": [
            {"title": "HR Business Partner", "company": "Emirates NBD", "duration": "2019 - Present",
             "description": "Lead full-cycle recruitment for retail banking division, 150+ hires annually."},
            {"title": "Talent Acquisition Specialist", "company": "du Telecom", "duration": "2017 - 2019",
             "description": "Sourced technical and non-technical candidates across UAE."}
        ],
        "certifications": [],
        "projects": [],
        "links": []
    }, "standard_chronological"))

# ---------- 6-8: Functional / skills-based resumes ----------
examples.append(rec(
    """MICHAEL TORRES
Skills-focused profile

TECHNICAL SKILLS
- Machine Learning: scikit-learn, PyTorch, XGBoost
- Data Engineering: Airflow, Spark, dbt
- Cloud: GCP, BigQuery

CAREER HISTORY (unstructured)
Worked at several startups over the last 5 years building data pipelines and ML models,
including a 2-year stint at a fintech startup and 1.5 years at an e-commerce analytics company.

No formal degree listed; self-taught via online courses and bootcamps.""",
    {
        "name": "Michael Torres",
        "email": None,
        "phone": None,
        "location": None,
        "summary": None,
        "skills": ["Machine Learning", "scikit-learn", "PyTorch", "XGBoost", "Airflow", "Spark", "dbt", "GCP", "BigQuery"],
        "education": [],
        "experience": [
            {"title": None, "company": "fintech startup", "duration": "2 years", "description": None},
            {"title": None, "company": "e-commerce analytics company", "duration": "1.5 years", "description": None}
        ],
        "certifications": [],
        "projects": [],
        "links": []
    }, "functional_unstructured"))

examples.append(rec(
    """SARAH JENKINS
Freelance Graphic Designer

CAPABILITIES
Adobe Photoshop, Illustrator, InDesign, Figma, Brand Identity Design, Typography

I have designed brand identities for over 30 small businesses since going independent.
Previously worked in-house at a design agency for 3 years before freelancing.

Contact: sarahjenkins.design@gmail.com""",
    {
        "name": "Sarah Jenkins",
        "email": "sarahjenkins.design@gmail.com",
        "phone": None,
        "location": None,
        "summary": "Freelance Graphic Designer",
        "skills": ["Adobe Photoshop", "Illustrator", "InDesign", "Figma", "Brand Identity Design", "Typography"],
        "education": [],
        "experience": [
            {"title": "Freelance Graphic Designer", "company": None, "duration": None,
             "description": "Designed brand identities for over 30 small businesses since going independent."},
            {"title": "Designer", "company": "design agency", "duration": "3 years",
             "description": "In-house role prior to freelancing."}
        ],
        "certifications": [],
        "projects": [],
        "links": []
    }, "functional_unstructured"))

examples.append(rec(
    """RAJESH KUMAR
Skills: Tally ERP, GST Filing, Bookkeeping, MS Excel, Bank Reconciliation

Over 8 years handling accounts for small and medium retail businesses in Jaipur.
Currently freelancing as an accounting consultant for 4 regular clients.""",
    {
        "name": "Rajesh Kumar",
        "email": None,
        "phone": None,
        "location": "Jaipur",
        "summary": "Accounting consultant with over 8 years handling accounts for small and medium retail businesses.",
        "skills": ["Tally ERP", "GST Filing", "Bookkeeping", "MS Excel", "Bank Reconciliation"],
        "education": [],
        "experience": [
            {"title": "Accounting Consultant", "company": "Freelance", "duration": None,
             "description": "Handling accounts for 4 regular retail business clients."}
        ],
        "certifications": [],
        "projects": [],
        "links": []
    }, "functional_unstructured"))

# ---------- 9-11: Academic CV with publications ----------
examples.append(rec(
    """Dr. Elena Petrova
Postdoctoral Researcher, Department of Physics, MIT
elena.petrova@mit.edu

EDUCATION
PhD in Theoretical Physics, ETH Zurich, 2020
MSc in Physics, Moscow State University, 2016

RESEARCH INTERESTS
Condensed matter theory, quantum computing hardware

PUBLICATIONS
- Petrova, E. et al. "Topological states in engineered lattices." Nature Physics, 2022.
- Petrova, E., Muller, J. "Decoherence in superconducting qubits." PRL, 2021.

CURRENT POSITION
Postdoctoral Fellow, MIT, 2020 - Present
Leading a research group on scalable qubit architectures.""",
    {
        "name": "Elena Petrova",
        "email": "elena.petrova@mit.edu",
        "phone": None,
        "location": None,
        "summary": None,
        "skills": ["Condensed matter theory", "Quantum computing hardware"],
        "education": [
            {"degree": "PhD in Theoretical Physics", "institution": "ETH Zurich", "year": "2020"},
            {"degree": "MSc in Physics", "institution": "Moscow State University", "year": "2016"}
        ],
        "experience": [
            {"title": "Postdoctoral Fellow", "company": "MIT", "duration": "2020 - Present",
             "description": "Leading a research group on scalable qubit architectures."}
        ],
        "certifications": [],
        "projects": [],
        "links": []
    }, "academic_cv"))

examples.append(rec(
    """AMIT VERMA, PhD (Candidate)
PhD Candidate, Computer Science, IIT Delhi
amit.verma@iitd.ac.in

EDUCATION
PhD Computer Science (in progress, expected 2027), IIT Delhi
M.Tech Computer Science, IIT Bombay, 2021
B.Tech Computer Science, NIT Trichy, 2019

TEACHING
Teaching Assistant for Machine Learning course, 2022-2024

No industry work experience listed.""",
    {
        "name": "Amit Verma",
        "email": "amit.verma@iitd.ac.in",
        "phone": None,
        "location": None,
        "summary": None,
        "skills": [],
        "education": [
            {"degree": "PhD Computer Science (in progress)", "institution": "IIT Delhi", "year": "Expected 2027"},
            {"degree": "M.Tech Computer Science", "institution": "IIT Bombay", "year": "2021"},
            {"degree": "B.Tech Computer Science", "institution": "NIT Trichy", "year": "2019"}
        ],
        "experience": [
            {"title": "Teaching Assistant", "company": "IIT Delhi", "duration": "2022-2024",
             "description": "Teaching Assistant for Machine Learning course."}
        ],
        "certifications": [],
        "projects": [],
        "links": []
    }, "academic_cv"))

examples.append(rec(
    """Prof. Linda Osei
Associate Professor of Economics, University of Ghana

EDUCATION
PhD Economics, London School of Economics, 2012
MSc Economics, University of Ghana, 2007

SELECTED PUBLICATIONS (12 total, showing 2)
1. Osei, L. "Informal Labor Markets in West Africa." Journal of Development Economics, 2019.
2. Osei, L., Boateng, K. "Microfinance and Poverty Reduction." World Development, 2016.

APPOINTMENTS
Associate Professor, University of Ghana, 2018 - Present
Assistant Professor, University of Ghana, 2012 - 2018""",
    {
        "name": "Linda Osei",
        "email": None,
        "phone": None,
        "location": None,
        "summary": None,
        "skills": [],
        "education": [
            {"degree": "PhD Economics", "institution": "London School of Economics", "year": "2012"},
            {"degree": "MSc Economics", "institution": "University of Ghana", "year": "2007"}
        ],
        "experience": [
            {"title": "Associate Professor", "company": "University of Ghana", "duration": "2018 - Present", "description": None},
            {"title": "Assistant Professor", "company": "University of Ghana", "duration": "2012 - 2018", "description": None}
        ],
        "certifications": [],
        "projects": [],
        "links": []
    }, "academic_cv"))

# ---------- 12-13: Missing email ----------
examples.append(rec(
    """Grace Adeyemi
Phone: 070-1234-5678 | Lagos, Nigeria

Customer support lead with experience managing teams of 10+.

Skills: Zendesk, Team Leadership, Conflict Resolution, KPI Reporting

Experience
Customer Support Lead, Paystack, 2020 - Present
Support Agent, Jumia, 2018 - 2020""",
    {
        "name": "Grace Adeyemi",
        "email": None,
        "phone": "070-1234-5678",
        "location": "Lagos, Nigeria",
        "summary": "Customer support lead with experience managing teams of 10+.",
        "skills": ["Zendesk", "Team Leadership", "Conflict Resolution", "KPI Reporting"],
        "education": [],
        "experience": [
            {"title": "Customer Support Lead", "company": "Paystack", "duration": "2020 - Present", "description": None},
            {"title": "Support Agent", "company": "Jumia", "duration": "2018 - 2020", "description": None}
        ],
        "certifications": [],
        "projects": [],
        "links": []
    }, "missing_email"))

examples.append(rec(
    """TOM HARRIS
Warehouse Supervisor | Manchester, UK | Tel: 07911 123456

10 years' experience in warehouse operations and logistics.
Skills: Forklift Certified, Inventory Management, SAP WM, Team Supervision

Warehouse Supervisor, DHL, 2015 - Present
Warehouse Operative, Amazon, 2012 - 2015""",
    {
        "name": "Tom Harris",
        "email": None,
        "phone": "07911 123456",
        "location": "Manchester, UK",
        "summary": "10 years' experience in warehouse operations and logistics.",
        "skills": ["Forklift Certified", "Inventory Management", "SAP WM", "Team Supervision"],
        "education": [],
        "experience": [
            {"title": "Warehouse Supervisor", "company": "DHL", "duration": "2015 - Present", "description": None},
            {"title": "Warehouse Operative", "company": "Amazon", "duration": "2012 - 2015", "description": None}
        ],
        "certifications": [],
        "projects": [],
        "links": []
    }, "missing_email"))

# ---------- 14-15: Missing phone ----------
examples.append(rec(
    """Nina Kowalski
nina.kowalski@protonmail.com | Warsaw, Poland

UX researcher passionate about accessible design.

Skills: User Interviews, Usability Testing, Figma, Miro

Experience
UX Researcher, Allegro, 2021 - Present
UX Research Intern, CD Projekt Red, 2020 - 2021

Education
MA in Cognitive Science, University of Warsaw, 2020""",
    {
        "name": "Nina Kowalski",
        "email": "nina.kowalski@protonmail.com",
        "phone": None,
        "location": "Warsaw, Poland",
        "summary": "UX researcher passionate about accessible design.",
        "skills": ["User Interviews", "Usability Testing", "Figma", "Miro"],
        "education": [{"degree": "MA in Cognitive Science", "institution": "University of Warsaw", "year": "2020"}],
        "experience": [
            {"title": "UX Researcher", "company": "Allegro", "duration": "2021 - Present", "description": None},
            {"title": "UX Research Intern", "company": "CD Projekt Red", "duration": "2020 - 2021", "description": None}
        ],
        "certifications": [],
        "projects": [],
        "links": []
    }, "missing_phone"))

examples.append(rec(
    """Carlos Mendez
carlos.mendez.eng@gmail.com
Mexico City, Mexico

Mechanical engineer with automotive industry background.

Skills: SolidWorks, AutoCAD, GD&T, Six Sigma

Experience
Mechanical Design Engineer, Nissan Mexicana, 2019 - Present
Junior Engineer, Bosch, 2017 - 2019

Education
BS Mechanical Engineering, Tecnologico de Monterrey, 2017""",
    {
        "name": "Carlos Mendez",
        "email": "carlos.mendez.eng@gmail.com",
        "phone": None,
        "location": "Mexico City, Mexico",
        "summary": "Mechanical engineer with automotive industry background.",
        "skills": ["SolidWorks", "AutoCAD", "GD&T", "Six Sigma"],
        "education": [{"degree": "BS Mechanical Engineering", "institution": "Tecnologico de Monterrey", "year": "2017"}],
        "experience": [
            {"title": "Mechanical Design Engineer", "company": "Nissan Mexicana", "duration": "2019 - Present", "description": None},
            {"title": "Junior Engineer", "company": "Bosch", "duration": "2017 - 2019", "description": None}
        ],
        "certifications": [],
        "projects": [],
        "links": []
    }, "missing_phone"))

# ---------- 16-18: Missing dates/duration ----------
examples.append(rec(
    """Wei Zhang
wei.zhang88@qq.com | Shanghai, China

Software Engineer at Tencent working on WeChat Pay backend systems.
Previously at Alibaba on the Taobao search team.

Skills: Java, Kafka, Redis, MySQL""",
    {
        "name": "Wei Zhang",
        "email": "wei.zhang88@qq.com",
        "phone": None,
        "location": "Shanghai, China",
        "summary": None,
        "skills": ["Java", "Kafka", "Redis", "MySQL"],
        "education": [],
        "experience": [
            {"title": "Software Engineer", "company": "Tencent", "duration": None,
             "description": "Working on WeChat Pay backend systems."},
            {"title": None, "company": "Alibaba", "duration": None,
             "description": "On the Taobao search team."}
        ],
        "certifications": [],
        "projects": [],
        "links": []
    }, "missing_dates"))

examples.append(rec(
    """Isabella Rossi
Rome, Italy | isabella.rossi@libero.it

Pastry chef trained in classic French technique. Worked at Le Cinque Terre restaurant
and later opened a small patisserie. Also did a stage at a Michelin-starred kitchen in Paris.

Skills: Viennoiserie, Chocolate Work, Sugar Art, Menu Development""",
    {
        "name": "Isabella Rossi",
        "email": "isabella.rossi@libero.it",
        "phone": None,
        "location": "Rome, Italy",
        "summary": "Pastry chef trained in classic French technique.",
        "skills": ["Viennoiserie", "Chocolate Work", "Sugar Art", "Menu Development"],
        "education": [],
        "experience": [
            {"title": "Pastry Chef", "company": "Le Cinque Terre restaurant", "duration": None, "description": None},
            {"title": "Owner/Pastry Chef", "company": "own patisserie", "duration": None, "description": None},
            {"title": "Stagiaire", "company": "Michelin-starred kitchen, Paris", "duration": None, "description": None}
        ],
        "certifications": [],
        "projects": [],
        "links": []
    }, "missing_dates"))

examples.append(rec(
    """Ben Okafor
Software developer. Worked at a few startups including one in fintech and one in edtech.
Comfortable with React and Django. Based remotely, open to relocation.
Email: ben.okafor.dev@gmail.com""",
    {
        "name": "Ben Okafor",
        "email": "ben.okafor.dev@gmail.com",
        "phone": None,
        "location": None,
        "summary": "Software developer open to relocation, works remotely.",
        "skills": ["React", "Django"],
        "education": [],
        "experience": [
            {"title": "Software Developer", "company": "fintech startup", "duration": None, "description": None},
            {"title": "Software Developer", "company": "edtech startup", "duration": None, "description": None}
        ],
        "certifications": [],
        "projects": [],
        "links": []
    }, "missing_dates"))

# ---------- 19-21: No work experience (fresher) ----------
examples.append(rec(
    """Sneha Iyer
sneha.iyer2024@gmail.com | +91-9123456780 | Pune, India

Final year B.E. student in Computer Engineering, graduating June 2025.
Seeking entry-level software development roles.

Skills: Java, Python, DSA, SQL, Git

Education
B.E. Computer Engineering, Pune Institute of Computer Technology, 2021 - 2025 (expected)

Projects
- Built a food delivery app clone using React Native and Firebase.
- Implemented a pathfinding visualizer in Python.""",
    {
        "name": "Sneha Iyer",
        "email": "sneha.iyer2024@gmail.com",
        "phone": "+91-9123456780",
        "location": "Pune, India",
        "summary": "Final year B.E. student in Computer Engineering, graduating June 2025, seeking entry-level software development roles.",
        "skills": ["Java", "Python", "DSA", "SQL", "Git"],
        "education": [{"degree": "B.E. Computer Engineering", "institution": "Pune Institute of Computer Technology", "year": "2021 - 2025 (expected)"}],
        "experience": [],
        "certifications": [],
        "projects": [
            {"name": "Food delivery app clone", "description": "Built using React Native and Firebase."},
            {"name": "Pathfinding visualizer", "description": "Implemented in Python."}
        ],
        "links": []
    }, "fresher_no_experience"))

examples.append(rec(
    """Marcus Lee
High school graduate applying for entry-level retail positions.
Contact: marcus.lee.99@gmail.com, 555-0173

Skills: Customer Service, Cash Handling, Punctual, Team Player

Education
High School Diploma, Lincoln High School, 2024""",
    {
        "name": "Marcus Lee",
        "email": "marcus.lee.99@gmail.com",
        "phone": "555-0173",
        "location": None,
        "summary": "High school graduate applying for entry-level retail positions.",
        "skills": ["Customer Service", "Cash Handling", "Punctual", "Team Player"],
        "education": [{"degree": "High School Diploma", "institution": "Lincoln High School", "year": "2024"}],
        "experience": [],
        "certifications": [],
        "projects": [],
        "links": []
    }, "fresher_no_experience"))

examples.append(rec(
    """Aditi Sharma
B.Sc. Statistics graduate, 2025. No prior work experience.
Interested in data analyst roles. aditi.sharma.stats@gmail.com

Skills: R, Excel, SQL, Power BI

Education: B.Sc. Statistics, Delhi University, 2022-2025""",
    {
        "name": "Aditi Sharma",
        "email": "aditi.sharma.stats@gmail.com",
        "phone": None,
        "location": None,
        "summary": "B.Sc. Statistics graduate interested in data analyst roles.",
        "skills": ["R", "Excel", "SQL", "Power BI"],
        "education": [{"degree": "B.Sc. Statistics", "institution": "Delhi University", "year": "2022-2025"}],
        "experience": [],
        "certifications": [],
        "projects": [],
        "links": []
    }, "fresher_no_experience"))

# ---------- 22-23: Career gap ----------
examples.append(rec(
    """Rebecca Nolan
rebecca.nolan@gmail.com | Austin, TX

Marketing professional returning to the workforce after a 3-year career break for
full-time caregiving. Prior to the break, 6 years in brand marketing.

Skills: Brand Strategy, Campaign Management, Adobe Creative Suite

Experience
Brand Manager, Whole Foods Market, 2015 - 2021
Career break (caregiving), 2021 - 2024""",
    {
        "name": "Rebecca Nolan",
        "email": "rebecca.nolan@gmail.com",
        "phone": None,
        "location": "Austin, TX",
        "summary": "Marketing professional returning to the workforce after a 3-year career break for full-time caregiving. 6 years prior experience in brand marketing.",
        "skills": ["Brand Strategy", "Campaign Management", "Adobe Creative Suite"],
        "education": [],
        "experience": [
            {"title": "Brand Manager", "company": "Whole Foods Market", "duration": "2015 - 2021", "description": None},
            {"title": "Career Break (Caregiving)", "company": None, "duration": "2021 - 2024", "description": None}
        ],
        "certifications": [],
        "projects": [],
        "links": []
    }, "career_gap"))

examples.append(rec(
    """Samuel Osei
samuel.osei@gmail.com | Accra, Ghana

Civil engineer, took 2 years off (2020-2022) to travel and pursue personal projects,
now seeking to re-enter engineering consulting.

Experience
Site Engineer, Julius Berger, 2015 - 2020
(Gap 2020 - 2022)

Skills: AutoCAD, Structural Analysis, Project Scheduling""",
    {
        "name": "Samuel Osei",
        "email": "samuel.osei@gmail.com",
        "phone": None,
        "location": "Accra, Ghana",
        "summary": "Civil engineer seeking to re-enter engineering consulting after a 2-year career break.",
        "skills": ["AutoCAD", "Structural Analysis", "Project Scheduling"],
        "education": [],
        "experience": [
            {"title": "Site Engineer", "company": "Julius Berger", "duration": "2015 - 2020", "description": None},
            {"title": "Career Break", "company": None, "duration": "2020 - 2022", "description": None}
        ],
        "certifications": [],
        "projects": [],
        "links": []
    }, "career_gap"))

# ---------- 24-25: Multiple degrees ----------
examples.append(rec(
    """Dr. Kevin Park
kevin.park.md@gmail.com | Seoul, South Korea

Cardiologist with combined clinical and research background.

Education
MD, Seoul National University College of Medicine, 2010
PhD in Biomedical Engineering, KAIST, 2015
Residency in Internal Medicine, Samsung Medical Center, 2013
Fellowship in Cardiology, Asan Medical Center, 2017

Experience
Attending Cardiologist, Asan Medical Center, 2017 - Present""",
    {
        "name": "Kevin Park",
        "email": "kevin.park.md@gmail.com",
        "phone": None,
        "location": "Seoul, South Korea",
        "summary": "Cardiologist with combined clinical and research background.",
        "skills": [],
        "education": [
            {"degree": "MD", "institution": "Seoul National University College of Medicine", "year": "2010"},
            {"degree": "PhD in Biomedical Engineering", "institution": "KAIST", "year": "2015"},
            {"degree": "Residency in Internal Medicine", "institution": "Samsung Medical Center", "year": "2013"},
            {"degree": "Fellowship in Cardiology", "institution": "Asan Medical Center", "year": "2017"}
        ],
        "experience": [
            {"title": "Attending Cardiologist", "company": "Asan Medical Center", "duration": "2017 - Present", "description": None}
        ],
        "certifications": [],
        "projects": [],
        "links": []
    }, "multiple_degrees"))

examples.append(rec(
    """Julia Fernandez
julia.fernandez@gmail.com | Sao Paulo, Brazil

Lawyer specializing in international trade law.

Education
LLM International Law, Georgetown University, 2018
Bachelor of Laws (LLB), Universidade de Sao Paulo, 2015
Postgraduate Certificate in Arbitration, FGV Law School, 2019

Experience
Senior Associate, Pinheiro Neto Advogados, 2019 - Present
Associate, Mattos Filho, 2015 - 2018""",
    {
        "name": "Julia Fernandez",
        "email": "julia.fernandez@gmail.com",
        "phone": None,
        "location": "Sao Paulo, Brazil",
        "summary": "Lawyer specializing in international trade law.",
        "skills": [],
        "education": [
            {"degree": "LLM International Law", "institution": "Georgetown University", "year": "2018"},
            {"degree": "Bachelor of Laws (LLB)", "institution": "Universidade de Sao Paulo", "year": "2015"},
            {"degree": "Postgraduate Certificate in Arbitration", "institution": "FGV Law School", "year": "2019"}
        ],
        "experience": [
            {"title": "Senior Associate", "company": "Pinheiro Neto Advogados", "duration": "2019 - Present", "description": None},
            {"title": "Associate", "company": "Mattos Filho", "duration": "2015 - 2018", "description": None}
        ],
        "certifications": [],
        "projects": [],
        "links": []
    }, "multiple_degrees"))

# ---------- 26-28: Non-standard section headers ----------
examples.append(rec(
    """VIKRAM SINGH
vikram.singh@gmail.com | Jaipur, India

WHERE I'VE WORKED
Product Designer, Swiggy, 2021 - Present
Design Intern, CRED, 2020 - 2021

STUFF I'M GOOD AT
Figma, User Research, Design Systems, Prototyping

MY BACKGROUND
B.Des, National Institute of Design, 2020""",
    {
        "name": "Vikram Singh",
        "email": "vikram.singh@gmail.com",
        "phone": None,
        "location": "Jaipur, India",
        "summary": None,
        "skills": ["Figma", "User Research", "Design Systems", "Prototyping"],
        "education": [{"degree": "B.Des", "institution": "National Institute of Design", "year": "2020"}],
        "experience": [
            {"title": "Product Designer", "company": "Swiggy", "duration": "2021 - Present", "description": None},
            {"title": "Design Intern", "company": "CRED", "duration": "2020 - 2021", "description": None}
        ],
        "certifications": [],
        "projects": [],
        "links": []
    }, "nonstandard_headers"))

examples.append(rec(
    """EMMA WATSON-CLARKE
emma.wc@gmail.com

MY JOURNEY SO FAR
2022-Present: Content Strategist at BBC
2019-2022: Journalist at The Guardian

WHAT I BRING
Storytelling, SEO Writing, Editorial Planning, WordPress

CREDENTIALS
BA Journalism, Cardiff University, 2019""",
    {
        "name": "Emma Watson-Clarke",
        "email": "emma.wc@gmail.com",
        "phone": None,
        "location": None,
        "summary": None,
        "skills": ["Storytelling", "SEO Writing", "Editorial Planning", "WordPress"],
        "education": [{"degree": "BA Journalism", "institution": "Cardiff University", "year": "2019"}],
        "experience": [
            {"title": "Content Strategist", "company": "BBC", "duration": "2022-Present", "description": None},
            {"title": "Journalist", "company": "The Guardian", "duration": "2019-2022", "description": None}
        ],
        "certifications": [],
        "projects": [],
        "links": []
    }, "nonstandard_headers"))

examples.append(rec(
    """RYAN COOPER
ryan.cooper@gmail.com

THE TOOLKIT
Salesforce, Excel, Power BI, SQL

BATTLES FOUGHT
Sales Operations Analyst, Oracle, 2020 - Present
Reduced quote-to-cash cycle time by 30%.

Sales Analyst, HubSpot, 2018 - 2020

SCHOOLING
BBA, Indiana University, 2018""",
    {
        "name": "Ryan Cooper",
        "email": "ryan.cooper@gmail.com",
        "phone": None,
        "location": None,
        "summary": None,
        "skills": ["Salesforce", "Excel", "Power BI", "SQL"],
        "education": [{"degree": "BBA", "institution": "Indiana University", "year": "2018"}],
        "experience": [
            {"title": "Sales Operations Analyst", "company": "Oracle", "duration": "2020 - Present",
             "description": "Reduced quote-to-cash cycle time by 30%."},
            {"title": "Sales Analyst", "company": "HubSpot", "duration": "2018 - 2020", "description": None}
        ],
        "certifications": [],
        "projects": [],
        "links": []
    }, "nonstandard_headers"))

# ---------- 29-31: OCR-noisy / garbled text ----------
examples.append(rec(
    """5uresh Patel
5uresh.pate1@gmai1.com | Ahm3dabad, lndia

50ftware Enginee r with 5 y3ars 0f experi3nce in java devel0pment.

5ki11s: Java, 5pring B0ot, MySQ1, J1RA

Exp3rience
5enior S0ftware Eng1neer, lnfosys, 2019 - Pr3sent
5oftware Engin3er, TCS, 2017 - 2019""",
    {
        "name": "Suresh Patel",
        "email": "suresh.patel@gmail.com",
        "phone": None,
        "location": "Ahmedabad, India",
        "summary": "Software Engineer with 5 years of experience in java development.",
        "skills": ["Java", "Spring Boot", "MySQL", "JIRA"],
        "education": [],
        "experience": [
            {"title": "Senior Software Engineer", "company": "Infosys", "duration": "2019 - Present", "description": None},
            {"title": "Software Engineer", "company": "TCS", "duration": "2017 - 2019", "description": None}
        ],
        "certifications": [],
        "projects": [],
        "links": []
    }, "ocr_noisy"))

examples.append(rec(
    """MARlA G0NZALEZ
mari a.gonza1ez@gmai1 . com  |  Madrid , 5pain

Financiaf Ana1yst wlth exp0sure t0 equity researc h.

5kills : Exce1, Bl0omberg Termina1, Financia1 M0deling

Exp erience
Financia1 Anaiyst , 5antander , 2020-Presemt""",
    {
        "name": "Maria Gonzalez",
        "email": "maria.gonzalez@gmail.com",
        "phone": None,
        "location": "Madrid, Spain",
        "summary": "Financial Analyst with exposure to equity research.",
        "skills": ["Excel", "Bloomberg Terminal", "Financial Modeling"],
        "education": [],
        "experience": [
            {"title": "Financial Analyst", "company": "Santander", "duration": "2020-Present", "description": None}
        ],
        "certifications": [],
        "projects": [],
        "links": []
    }, "ocr_noisy"))

examples.append(rec(
    """Ah m3d Ha ssan
ahm3d.h @outl0ok . com

Civi1 5ite Supervis0r , Ca ir0 , Egypt

5ki11s : AutoCAD , 5ite Managem3nt , 5af3ty C0mpliance

W0rk
5ite 5upervis0r , 0rasc0m C0nstructi0n , 2018 - Present""",
    {
        "name": "Ahmed Hassan",
        "email": "ahmed.h@outlook.com",
        "phone": None,
        "location": "Cairo, Egypt",
        "summary": None,
        "skills": ["AutoCAD", "Site Management", "Safety Compliance"],
        "education": [],
        "experience": [
            {"title": "Site Supervisor", "company": "Orascom Construction", "duration": "2018 - Present", "description": None}
        ],
        "certifications": [],
        "projects": [],
        "links": []
    }, "ocr_noisy"))

# ---------- 32-34: Table/column layout flattened to text ----------
examples.append(rec(
    """Name Olivia Bennett Email olivia.bennett@gmail.com Phone 555-2231 Location Denver CO
Skills Python | SQL | Tableau | Excel
Experience Data Analyst Trailblaze Inc 2021-Present Junior Analyst BrightPath Co 2019-2021
Education BS Data Science University of Colorado 2019""",
    {
        "name": "Olivia Bennett",
        "email": "olivia.bennett@gmail.com",
        "phone": "555-2231",
        "location": "Denver, CO",
        "summary": None,
        "skills": ["Python", "SQL", "Tableau", "Excel"],
        "education": [{"degree": "BS Data Science", "institution": "University of Colorado", "year": "2019"}],
        "experience": [
            {"title": "Data Analyst", "company": "Trailblaze Inc", "duration": "2021-Present", "description": None},
            {"title": "Junior Analyst", "company": "BrightPath Co", "duration": "2019-2021", "description": None}
        ],
        "certifications": [],
        "projects": [],
        "links": []
    }, "table_flattened"))

examples.append(rec(
    """Name: Hiroshi Tanaka | Location: Tokyo, Japan | Email: h.tanaka@gmail.com
Role | Company | Years
Mechanical Design Lead | Toyota | 2018-Present
Design Engineer | Honda | 2014-2018
Skills: CATIA, Six Sigma, DFMEA, Kaizen
Education: BEng Mechanical Engineering | Tokyo Institute of Technology | 2014""",
    {
        "name": "Hiroshi Tanaka",
        "email": "h.tanaka@gmail.com",
        "phone": None,
        "location": "Tokyo, Japan",
        "summary": None,
        "skills": ["CATIA", "Six Sigma", "DFMEA", "Kaizen"],
        "education": [{"degree": "BEng Mechanical Engineering", "institution": "Tokyo Institute of Technology", "year": "2014"}],
        "experience": [
            {"title": "Mechanical Design Lead", "company": "Toyota", "duration": "2018-Present", "description": None},
            {"title": "Design Engineer", "company": "Honda", "duration": "2014-2018", "description": None}
        ],
        "certifications": [],
        "projects": [],
        "links": []
    }, "table_flattened"))

examples.append(rec(
    """Candidate: Laura Schmidt Contact: laura.schmidt@gmx.de Berlin Germany
Position Company Duration
Product Owner Zalando 2020-Present
Business Analyst SAP 2016-2020
Skills Scrum JIRA Confluence Roadmapping
Degree MSc Business Informatics TU Berlin 2016""",
    {
        "name": "Laura Schmidt",
        "email": "laura.schmidt@gmx.de",
        "phone": None,
        "location": "Berlin, Germany",
        "summary": None,
        "skills": ["Scrum", "JIRA", "Confluence", "Roadmapping"],
        "education": [{"degree": "MSc Business Informatics", "institution": "TU Berlin", "year": "2016"}],
        "experience": [
            {"title": "Product Owner", "company": "Zalando", "duration": "2020-Present", "description": None},
            {"title": "Business Analyst", "company": "SAP", "duration": "2016-2020", "description": None}
        ],
        "certifications": [],
        "projects": [],
        "links": []
    }, "table_flattened"))

# ---------- 35-36: Bullet-fragment style, no full sentences ----------
examples.append(rec(
    """JAMES WALKER
james.walker@gmail.com

- 6 yrs DevOps
- AWS, Terraform, Jenkins, Ansible
- Reduced deploy time 45min -> 5min
- On-call rotation lead
- Ex-Netflix, Ex-Spotify
- No degree, self-taught""",
    {
        "name": "James Walker",
        "email": "james.walker@gmail.com",
        "phone": None,
        "location": None,
        "summary": "6 years DevOps experience.",
        "skills": ["AWS", "Terraform", "Jenkins", "Ansible"],
        "education": [],
        "experience": [
            {"title": None, "company": "Netflix", "duration": None,
             "description": "Reduced deploy time from 45min to 5min. On-call rotation lead."},
            {"title": None, "company": "Spotify", "duration": None, "description": None}
        ],
        "certifications": [],
        "projects": [],
        "links": []
    }, "bullet_fragments"))

examples.append(rec(
    """PRIYANKA DESAI
priyanka.d@gmail.com | Mumbai

- QA Engineer
- Selenium / Cypress / Postman
- 4 yrs testing fintech apps
- ISTQB certified
- Prev: PayU, Current: Razorpay""",
    {
        "name": "Priyanka Desai",
        "email": "priyanka.d@gmail.com",
        "phone": None,
        "location": "Mumbai",
        "summary": "QA Engineer with 4 years testing fintech apps.",
        "skills": ["Selenium", "Cypress", "Postman"],
        "education": [],
        "experience": [
            {"title": "QA Engineer", "company": "Razorpay", "duration": "Current", "description": None},
            {"title": "QA Engineer", "company": "PayU", "duration": "Previous", "description": None}
        ],
        "certifications": ["ISTQB"],
        "projects": [],
        "links": []
    }, "bullet_fragments"))

# ---------- 37-38: Multiple phone numbers/emails ----------
examples.append(rec(
    """Anna Kim
Personal: anna.kim.personal@gmail.com | Work: anna.kim@company.com
Mobile: +82-10-1234-5678 | Office: +82-2-9876-5432
Seoul, South Korea

Product Manager, Coupang, 2020 - Present

Skills: Product Strategy, A/B Testing, SQL, JIRA""",
    {
        "name": "Anna Kim",
        "email": "anna.kim.personal@gmail.com",
        "phone": "+82-10-1234-5678",
        "location": "Seoul, South Korea",
        "summary": None,
        "skills": ["Product Strategy", "A/B Testing", "SQL", "JIRA"],
        "education": [],
        "experience": [
            {"title": "Product Manager", "company": "Coupang", "duration": "2020 - Present", "description": None}
        ],
        "certifications": [],
        "projects": [],
        "links": []
    }, "multiple_contacts"))

examples.append(rec(
    """Peter Novak
peter.novak@gmail.com / p.novak@consulting-firm.com
Cell: +420 777 123 456, Home: +420 2 1234 5678
Prague, Czech Republic

Management Consultant, McKinsey & Company, 2019 - Present""",
    {
        "name": "Peter Novak",
        "email": "peter.novak@gmail.com",
        "phone": "+420 777 123 456",
        "location": "Prague, Czech Republic",
        "summary": None,
        "skills": [],
        "education": [],
        "experience": [
            {"title": "Management Consultant", "company": "McKinsey & Company", "duration": "2019 - Present", "description": None}
        ],
        "certifications": [],
        "projects": [],
        "links": []
    }, "multiple_contacts"))

# ---------- 39-41: LinkedIn/GitHub/portfolio links ----------
examples.append(rec(
    """Tyler Brooks
tyler.brooks@gmail.com | Portland, OR
Portfolio: tylerbrooks.design | LinkedIn: linkedin.com/in/tylerbrooks | Dribbble: dribbble.com/tylerb

Product Designer, Figma, 2021 - Present

Skills: Figma, Framer, Design Systems, Motion Design""",
    {
        "name": "Tyler Brooks",
        "email": "tyler.brooks@gmail.com",
        "phone": None,
        "location": "Portland, OR",
        "summary": None,
        "skills": ["Figma", "Framer", "Design Systems", "Motion Design"],
        "education": [],
        "experience": [
            {"title": "Product Designer", "company": "Figma", "duration": "2021 - Present", "description": None}
        ],
        "certifications": [],
        "projects": [],
        "links": ["tylerbrooks.design", "linkedin.com/in/tylerbrooks", "dribbble.com/tylerb"]
    }, "with_links"))

examples.append(rec(
    """Aisha Patel
aisha.patel.dev@gmail.com
github.com/aishap | linkedin.com/in/aishapatel

ML Engineer, 3 years experience.
Skills: PyTorch, TensorFlow, MLflow, Docker

Experience: ML Engineer, Zeta AI, 2022 - Present""",
    {
        "name": "Aisha Patel",
        "email": "aisha.patel.dev@gmail.com",
        "phone": None,
        "location": None,
        "summary": "ML Engineer, 3 years experience.",
        "skills": ["PyTorch", "TensorFlow", "MLflow", "Docker"],
        "education": [],
        "experience": [
            {"title": "ML Engineer", "company": "Zeta AI", "duration": "2022 - Present", "description": None}
        ],
        "certifications": [],
        "projects": [],
        "links": ["github.com/aishap", "linkedin.com/in/aishapatel"]
    }, "with_links"))

examples.append(rec(
    """Daniel Osei
daniel.osei.writes@gmail.com
Twitter: @danwrites | Substack: danosei.substack.com

Freelance technical writer, 5 years experience.
Skills: Technical Writing, API Documentation, Markdown, Docs-as-code""",
    {
        "name": "Daniel Osei",
        "email": "daniel.osei.writes@gmail.com",
        "phone": None,
        "location": None,
        "summary": "Freelance technical writer, 5 years experience.",
        "skills": ["Technical Writing", "API Documentation", "Markdown", "Docs-as-code"],
        "education": [],
        "experience": [],
        "certifications": [],
        "projects": [],
        "links": ["@danwrites", "danosei.substack.com"]
    }, "with_links"))

# ---------- 42-43: Certifications only, no formal degree ----------
examples.append(rec(
    """Chris Bailey
chris.bailey@gmail.com | Miami, FL

AWS-certified cloud engineer, no traditional 4-year degree.

Certifications: AWS Certified Solutions Architect - Professional, AWS Certified DevOps Engineer,
CompTIA Security+

Experience
Cloud Engineer, Rackspace, 2020 - Present
Support Technician, GoDaddy, 2017 - 2020""",
    {
        "name": "Chris Bailey",
        "email": "chris.bailey@gmail.com",
        "phone": None,
        "location": "Miami, FL",
        "summary": "AWS-certified cloud engineer, no traditional 4-year degree.",
        "skills": [],
        "education": [],
        "experience": [
            {"title": "Cloud Engineer", "company": "Rackspace", "duration": "2020 - Present", "description": None},
            {"title": "Support Technician", "company": "GoDaddy", "duration": "2017 - 2020", "description": None}
        ],
        "certifications": ["AWS Certified Solutions Architect - Professional", "AWS Certified DevOps Engineer", "CompTIA Security+"],
        "projects": [],
        "links": []
    }, "certs_no_degree"))

examples.append(rec(
    """Meera Nair
meera.nair@gmail.com | Kochi

Certified Scrum Master and Product Owner. Left college after 1 year, self-taught in Agile practices.

Certifications: CSM (Scrum Alliance), CSPO (Scrum Alliance)

Experience
Scrum Master, TCS, 2019 - Present""",
    {
        "name": "Meera Nair",
        "email": "meera.nair@gmail.com",
        "phone": None,
        "location": "Kochi",
        "summary": "Certified Scrum Master and Product Owner.",
        "skills": [],
        "education": [],
        "experience": [
            {"title": "Scrum Master", "company": "TCS", "duration": "2019 - Present", "description": None}
        ],
        "certifications": ["CSM (Scrum Alliance)", "CSPO (Scrum Alliance)"],
        "projects": [],
        "links": []
    }, "certs_no_degree"))

# ---------- 44-45: Freelancer/contractor with concurrent roles ----------
examples.append(rec(
    """Jordan Blake
jordan.blake@gmail.com | Remote

Independent software contractor, currently juggling 3 concurrent client engagements.

Skills: Python, Django, React, PostgreSQL

Current Engagements (all ongoing since 2023)
Backend Contractor - Client A (fintech startup)
Frontend Contractor - Client B (e-commerce)
Technical Advisor - Client C (early-stage SaaS)""",
    {
        "name": "Jordan Blake",
        "email": "jordan.blake@gmail.com",
        "phone": None,
        "location": "Remote",
        "summary": "Independent software contractor, currently juggling 3 concurrent client engagements.",
        "skills": ["Python", "Django", "React", "PostgreSQL"],
        "education": [],
        "experience": [
            {"title": "Backend Contractor", "company": "Client A (fintech startup)", "duration": "since 2023", "description": None},
            {"title": "Frontend Contractor", "company": "Client B (e-commerce)", "duration": "since 2023", "description": None},
            {"title": "Technical Advisor", "company": "Client C (early-stage SaaS)", "duration": "since 2023", "description": None}
        ],
        "certifications": [],
        "projects": [],
        "links": []
    }, "concurrent_roles"))

examples.append(rec(
    """Natasha Ivanova
natasha.ivanova@gmail.com

Freelance translator (Russian/English/French), working with 4 agencies simultaneously
since 2021: LinguaCorp, TransGlobal, WordBridge, and EuroTrans.

Skills: Legal Translation, Medical Translation, CAT Tools (Trados)""",
    {
        "name": "Natasha Ivanova",
        "email": "natasha.ivanova@gmail.com",
        "phone": None,
        "location": None,
        "summary": "Freelance translator (Russian/English/French).",
        "skills": ["Legal Translation", "Medical Translation", "CAT Tools (Trados)"],
        "education": [],
        "experience": [
            {"title": "Freelance Translator", "company": "LinguaCorp", "duration": "since 2021", "description": None},
            {"title": "Freelance Translator", "company": "TransGlobal", "duration": "since 2021", "description": None},
            {"title": "Freelance Translator", "company": "WordBridge", "duration": "since 2021", "description": None},
            {"title": "Freelance Translator", "company": "EuroTrans", "duration": "since 2021", "description": None}
        ],
        "certifications": [],
        "projects": [],
        "links": []
    }, "concurrent_roles"))

# ---------- 46-47: Objective statement only, no summary ----------
examples.append(rec(
    """Brian Foster
brian.foster@gmail.com | Nashville, TN

OBJECTIVE: To obtain a mechanical engineering position where I can apply my CAD skills
and grow into a design leadership role within 5 years.

Skills: SolidWorks, ANSYS, GD&T

Education: BS Mechanical Engineering, University of Tennessee, 2024""",
    {
        "name": "Brian Foster",
        "email": "brian.foster@gmail.com",
        "phone": None,
        "location": "Nashville, TN",
        "summary": "To obtain a mechanical engineering position where I can apply my CAD skills and grow into a design leadership role within 5 years.",
        "skills": ["SolidWorks", "ANSYS", "GD&T"],
        "education": [{"degree": "BS Mechanical Engineering", "institution": "University of Tennessee", "year": "2024"}],
        "experience": [],
        "certifications": [],
        "projects": [],
        "links": []
    }, "objective_only"))

examples.append(rec(
    """Yuki Sato
yuki.sato@gmail.com

Objective: Seeking a junior data analyst role to leverage my statistics background
and grow my SQL and visualization skills in a fast-paced environment.

Skills: SQL, Excel, Python (basic)

Education: BA Economics, Waseda University, 2024""",
    {
        "name": "Yuki Sato",
        "email": "yuki.sato@gmail.com",
        "phone": None,
        "location": None,
        "summary": "Seeking a junior data analyst role to leverage my statistics background and grow my SQL and visualization skills in a fast-paced environment.",
        "skills": ["SQL", "Excel", "Python (basic)"],
        "education": [{"degree": "BA Economics", "institution": "Waseda University", "year": "2024"}],
        "experience": [],
        "certifications": [],
        "projects": [],
        "links": []
    }, "objective_only"))

# ---------- 48: Very short/minimal info resume ----------
examples.append(rec(
    """Alex Turner. Plumber. 15 years experience. Call 555-0199.""",
    {
        "name": "Alex Turner",
        "email": None,
        "phone": "555-0199",
        "location": None,
        "summary": "Plumber, 15 years experience.",
        "skills": [],
        "education": [],
        "experience": [],
        "certifications": [],
        "projects": [],
        "links": []
    }, "minimal_info"))

# ---------- 49: Irrelevant personal info that should NOT be extracted into schema fields ----------
examples.append(rec(
    """Robert Klein
robert.klein@gmail.com | Vienna, Austria
Age: 34 | Marital Status: Married | Nationality: Austrian
[Photo attached]

Financial Controller, Erste Group, 2018 - Present

Skills: SAP FI/CO, IFRS, Financial Reporting

Education: MSc Finance, WU Vienna, 2014""",
    {
        "name": "Robert Klein",
        "email": "robert.klein@gmail.com",
        "phone": None,
        "location": "Vienna, Austria",
        "summary": None,
        "skills": ["SAP FI/CO", "IFRS", "Financial Reporting"],
        "education": [{"degree": "MSc Finance", "institution": "WU Vienna", "year": "2014"}],
        "experience": [
            {"title": "Financial Controller", "company": "Erste Group", "duration": "2018 - Present", "description": None}
        ],
        "certifications": [],
        "projects": [],
        "links": []
    }, "irrelevant_personal_info_excluded"))

# ---------- 50: Abbreviated date formats + salary mention (hallucination trap) ----------
examples.append(rec(
    """Steve Adams
steve.adams@gmail.com

Sales Executive, Jan'20 - Present, TechCorp
Account Manager, Mar'17 - Dec'19, SalesForce Inc

Desired salary: $90,000+. Available to start immediately.

Skills: Salesforce CRM, Negotiation, Lead Generation""",
    {
        "name": "Steve Adams",
        "email": "steve.adams@gmail.com",
        "phone": None,
        "location": None,
        "summary": None,
        "skills": ["Salesforce CRM", "Negotiation", "Lead Generation"],
        "education": [],
        "experience": [
            {"title": "Sales Executive", "company": "TechCorp", "duration": "Jan'20 - Present", "description": None},
            {"title": "Account Manager", "company": "SalesForce Inc", "duration": "Mar'17 - Dec'19", "description": None}
        ],
        "certifications": [],
        "projects": [],
        "links": []
    }, "abbreviated_dates_salary_trap"))

if __name__ == "__main__":
    # Strip internal bookkeeping field before writing the actual training file
    clean = []
    for e in examples:
        e2 = {k: v for k, v in e.items() if k != "_category"}
        clean.append(e2)

    with open("resume_parsing_dataset.jsonl", "w", encoding="utf-8") as f:
        for e in clean:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    # Also write a category breakdown for our own QA / stratified split reference
    from collections import Counter
    cats = Counter(e["_category"] for e in examples)
    with open("category_breakdown.json", "w", encoding="utf-8") as f:
        json.dump(dict(cats), f, indent=2)

    print(f"Wrote {len(clean)} examples to resume_parsing_dataset.jsonl")
    print("Category breakdown:", dict(cats))
