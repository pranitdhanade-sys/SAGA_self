<div align="center">

# 📦 SAGA_self
**A lightweight, self-contained Python implementation of the Saga Design Pattern.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![GitHub stars](https://img.shields.io/github/stars/pranitdhanade-sys/SAGA_self?style=for-the-badge&color=gold)](https://github.com/pranitdhanade-sys/SAGA_self/stargazers)
[![Maintained](https://img.shields.io/badge/Maintained%3F-yes-green.svg?style=for-the-badge)](https://github.com/pranitdhanade-sys/SAGA_self/graphs/commit-activity)

<p align="center">
  <a href="#-about-the-project">About</a> •
  <a href="#-tech-stack">Tech Stack</a> •
  <a href="#-features">Features</a> •
  <a href="#-getting-started">Getting Started</a> •
  <a href="#-project-structure">Structure</a>
</p>

</div>

---

## 📖 About the Project

**SAGA_self** is a focused exploration of distributed transaction management. In microservices, maintaining data consistency without a central database is hard. This project demonstrates how to use the **Saga Pattern** to manage long-running business processes through a series of local transactions and compensating actions.

> [!IMPORTANT]
> This implementation focuses on the **backward recovery** (compensating) strategy to ensure that even if a step fails, your system returns to a consistent state.

---

## 🛠️ Tech Stack

| Technology | Purpose |
| :--- | :--- |
| **Python** | Core Language |
| **PyTest** | Unit & Integration Testing |
| **Logging** | Transaction Observability |

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white)

---

## 🚀 Features

- ✅ **Atomic-like Consistency:** Ensures "all or nothing" logic across distributed steps.
- 🔄 **Automatic Compensation:** Built-in logic to trigger "undo" functions when a failure is detected.
- 📊 **State Tracking:** Clear visibility into which step of the Saga is currently executing.
- 🧪 **Testable Core:** Decoupled logic allowing for easy mocking of service failures.

---

## 🧠 The SAGA Logic

```mermaid
graph TD
    A[Start Saga] --> B[Step 1: Success]
    B --> C[Step 2: Success]
    C --> D{Step 3: Failure?}
    D -- Yes --> E[Compensate Step 2]
    E --> F[Compensate Step 1]
    F --> G[System Consistent / Aborted]
    D -- No --> H[Saga Completed]

🛠️ Getting Started
🔧 Prerequisites

    Python 3.8+

    pip (Python package manager)

📥 Installation
```
# Clone the repository
git clone [https://github.com/pranitdhanade-sys/SAGA_self.git](https://github.com/pranitdhanade-sys/SAGA_self.git)

# Navigate to the project
cd SAGA_self

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```



```
SAGA_self/
├── saga_self/
│   ├── __init__.py
│   ├── core.py       # Core Saga Coordinator logic
│   └── utils.py      # Helper functions for logging/state
├── tests/
│   └── test_core.py  # Validation of success & failure paths
├── requirements.txt
├── README.md
└── LICENSE
```