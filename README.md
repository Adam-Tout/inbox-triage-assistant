# Inbox Triage Assistant

A simple email clustering tool that groups your last 200 emails into actionable clusters and enables one-click archiving.

## Gmail API Setup Instructions

### 1. Enable Gmail API
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable the Gmail API for your project
4. Go to "APIs & Services" > "Library" > Search for "Gmail API" > Enable

### 2. Create Credentials
1. Go to "APIs & Services" > "Credentials"
2. Click "Create Credentials" > "OAuth 2.0 Client IDs"
3. Choose "Desktop application"
4. Download the JSON file and save it as `credentials.json` in this folder

### 3. Install Dependencies
```bash
pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client
```

### 4. First Run Setup
- Run `python inbox_triage.py` for the first time
- Follow the authentication flow in your browser
- Grant permissions to access your Gmail

## Features
- Clusters last 200 emails into actionable groups
- Shows descriptive cluster names
- One-click archive entire clusters
- Simple command-line interface

## Usage
```bash
python inbox_triage.py
```
