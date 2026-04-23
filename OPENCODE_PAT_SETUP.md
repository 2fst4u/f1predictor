# OpenCode PAT Setup Instructions

To allow the OpenCode agents (like the Planner) to automatically trigger worker agents by creating sub-issues and commenting `/oc-easy` or `/oc-hard`, you need to set up a GitHub Personal Access Token (PAT).

By default, GitHub prevents actions initiated by the `GITHUB_TOKEN` from triggering subsequent workflows (to prevent infinite loops). Using a PAT bypasses this restriction.

### Steps to configure:

1. **Create a Fine-Grained Personal Access Token (PAT)**
   - Go to your GitHub account settings -> Developer Settings -> Personal Access Tokens -> Fine-grained tokens.
   - Click **Generate new token**.
   - **Repository access**: Select the specific repository where you are running OpenCode.
   - **Permissions**: Grant the following **Repository Permissions**:
     - `Contents`: Read & Write
     - `Issues`: Read & Write
     - `Pull Requests`: Read & Write
     - `Workflows`: Read & Write

2. **Add the PAT to your Repository Secrets**
   - Go to your repository's settings -> Secrets and variables -> Actions.
   - Click **New repository secret**.
   - **Name**: `OPENCODE_PAT`
   - **Secret**: Paste the token you generated in step 1.
   - Click **Add secret**.

The updated workflows will automatically detect this secret and allow the planner agent to automatically trigger the easy and hard workers!