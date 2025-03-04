## Contributing to the Project

Thank you for your interest in contributing to this project! There are two main ways to contribute:

1. **Project Team Members**: If you are part of the project team, you will submit your work in the `submissions-team/` directory.
2. **Community Members**: If you are not a project team member but still want to contribute, you can submit your work in the `submissions-community/` directory.

## Contribution Guidelines

### 1. Verify Git Installation
Make sure that you have git installed by running the following command in your terminal.

```bash
git --version
```

### 2. Fork the Repository
First, you need to fork the repository to your GitHub account. You can do this by clicking the `Fork` button at the top right of the repository page.

### 3. Clone the Repository
After forking, clone your forked repository to your local machine using the following command:

```bash
git clone https://github.com/YOUR_USERNAME/PROJECT_NAME.git
```

Replace `YOUR_USERNAME` with your GitHub username and `PROJECT_NAME` with the name of the repository.

Navigate to the project directory:

```bash
cd PROJECT_NAME
```

### 4. Set Up a Virtual Environment
To ensure a clean and consistent environment, set up a virtual environment using either Python's built-in `venv` module or Anaconda.

#### Using Python Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate    # On Windows
```
To install dependencies, run:
```bash
pip install -r requirements.txt
```

#### Using Anaconda
```bash
conda create --name myenv python=3.12
conda activate myenv
```
To install dependencies, run:
```bash
pip install -r requirements.txt
```

### 5. Add Your Contributions
#### For Project Team Members
- Place your submission in the `submissions-team/` directory.
- Create a new folder with your name inside `submissions-team/`.
- Add your files inside your personal folder.

**Example Structure:**
```
submissions-team/
│── your-name/
│   ├── data-analysis.py
│   ├── app.py
│   ├── requirements.txt
```

#### For Community Members
- Place your contribution in the `submissions-community/` directory.
- Create a new folder with your name inside `submissions-community/`.
- Add your files inside your personal folder.

**Example Structure:**
```
submissions-community/
│── your-name/
│   ├── data-analysis.py
│   ├── app.py
│   ├── requirements.txt
```

### 6. Commit and Push Your Changes
After adding your files, commit your changes with a meaningful commit message:

```bash
git add .
git commit -m "Added my contribution"
git push origin your-branch-name
```

### 7. Create a Pull Request (PR)
Once your changes are pushed, go to the original repository on GitHub and follow these steps:

1. Click on the `Pull Requests` tab.
2. Click `New Pull Request`.
3. Select your forked repository and the branch you pushed your changes to (main).
4. Add a title and description for your PR.
5. Click `Create Pull Request`.

Your PR will be reviewed, and once approved, it will be merged into the project!

### 8. Keep Your Fork Updated
To keep your fork up-to-date with the original repository:

1. Make sure to sync your fork regualrly so that all changes made to the main branch on the SDS repo are reflected on your fork.
2. Run ```bash git pull``` to pull down the recent changes from your GitHub account to your local environment.

A complete tutorial on how to use Git & GitHub can be found on the SDS platform. Below is the link to our GitHub course:

https://community.superdatascience.com/c/intro-to-git-github/?preview=true

If you have any questions, feel free to reach out by opening an issue or sending me an email shaheer@superdatascience.com Happy coding!