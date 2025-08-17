#!/usr/bin/env python3
"""
Inbox Triage Assistant - Web Version
Flask web app for email clustering and archiving
"""

import os
import pickle
import base64
import re
import json
from typing import List, Dict, Tuple
from collections import defaultdict
from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np

# Gmail API scopes
SCOPES = ['https://www.googleapis.com/auth/gmail.modify']

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'inbox_triage_secret_key')

class InboxTriageAssistant:
    def __init__(self):
        self.service = None
        self.emails = []
        self.clusters = {}
        self.cluster_names = {}
        
    def authenticate(self):
        """Authenticate with Gmail API"""
        creds = None
        
        # Load existing token
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)
        
        # If no valid credentials, get new ones
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                # Try to get credentials from environment variable (Railway)
                google_creds = os.environ.get('GOOGLE_CREDENTIALS')
                if google_creds:
                    # Create credentials.json from environment variable
                    with open('credentials.json', 'w') as f:
                        f.write(google_creds)
                
                if not os.path.exists('credentials.json'):
                    return False
                
                # Use headless flow for Railway deployment
                flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
                # Use headless authentication
                creds = flow.run_console()
            
            # Save credentials for next run
            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)
        
        try:
            self.service = build('gmail', 'v1', credentials=creds)
            return True
        except Exception as e:
            return False
    
    def fetch_emails(self, max_emails=200):
        """Fetch last 200 emails from inbox"""
        try:
            # Get email IDs
            results = self.service.users().messages().list(
                userId='me', 
                labelIds=['INBOX'],
                maxResults=max_emails
            ).execute()
            
            messages = results.get('messages', [])
            if not messages:
                return False
            
            # Fetch full email details
            self.emails = []
            for message in messages:
                msg = self.service.users().messages().get(
                    userId='me', 
                    id=message['id'],
                    format='full'
                ).execute()
                
                email_data = self._parse_email(msg)
                if email_data:
                    self.emails.append(email_data)
            
            return True
            
        except HttpError as error:
            return False
    
    def _parse_email(self, msg):
        """Parse email message into structured data"""
        try:
            headers = msg['payload']['headers']
            
            # Extract basic info
            subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
            sender = next((h['value'] for h in headers if h['name'] == 'From'), 'Unknown')
            date = next((h['value'] for h in headers if h['name'] == 'Date'), '')
            
            # Extract body text
            body = self._extract_body(msg['payload'])
            
            return {
                'id': msg['id'],
                'subject': subject,
                'sender': sender,
                'date': date,
                'body': body,
                'snippet': msg.get('snippet', ''),
                'labels': msg.get('labelIds', [])
            }
        except Exception as e:
            return None
    
    def _extract_body(self, payload):
        """Extract text body from email payload"""
        if 'body' in payload and payload['body'].get('data'):
            return base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8', errors='ignore')
        
        if 'parts' in payload:
            for part in payload['parts']:
                if part['mimeType'] == 'text/plain':
                    if 'data' in part['body']:
                        return base64.urlsafe_b64decode(part['body']['data']).decode('utf-8', errors='ignore')
        
        return ""
    
    def cluster_emails(self, num_clusters=5):
        """Cluster emails using TF-IDF and K-means"""
        if not self.emails:
            return False
        
        # Prepare text for clustering
        texts = []
        for email in self.emails:
            text = f"{email['subject']} {email['sender']} {email['snippet']}"
            texts.append(text)
        
        # Vectorize text
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        X = vectorizer.fit_transform(texts)
        
        # Cluster
        kmeans = KMeans(n_clusters=min(num_clusters, len(self.emails)), random_state=42)
        clusters = kmeans.fit_predict(X)
        
        # Group emails by cluster
        self.clusters = defaultdict(list)
        for i, cluster_id in enumerate(clusters):
            self.clusters[cluster_id].append(self.emails[i])
        
        # Generate cluster names
        self.cluster_names = self._generate_cluster_names()
        
        return True
    
    def _generate_cluster_names(self):
        """Generate descriptive names for clusters"""
        names = {}
        
        for cluster_id, emails in self.clusters.items():
            # Analyze common patterns
            subjects = [email['subject'].lower() for email in emails]
            senders = [email['sender'].lower() for email in emails]
            
            # Find common words in subjects
            all_words = []
            for subject in subjects:
                words = re.findall(r'\b\w+\b', subject)
                all_words.extend(words)
            
            # Count word frequency
            word_count = defaultdict(int)
            for word in all_words:
                if len(word) > 3:  # Skip short words
                    word_count[word] += 1
            
            # Find most common word
            if word_count:
                most_common = max(word_count.items(), key=lambda x: x[1])[0]
                names[cluster_id] = f"{most_common.title()} ({len(emails)} emails)"
            else:
                # Fallback to sender domain
                domains = []
                for sender in senders:
                    domain = sender.split('@')[-1] if '@' in sender else sender
                    domains.append(domain)
                
                if domains:
                    most_common_domain = max(set(domains), key=domains.count)
                    names[cluster_id] = f"{most_common_domain} ({len(emails)} emails)"
                else:
                    names[cluster_id] = f"Cluster {cluster_id + 1} ({len(emails)} emails)"
        
        return names
    
    def archive_cluster(self, cluster_id):
        """Archive all emails in a cluster"""
        if cluster_id not in self.clusters:
            return False
        
        emails = self.clusters[cluster_id]
        
        try:
            # Remove INBOX label and add ARCHIVE label
            for email in emails:
                self.service.users().messages().modify(
                    userId='me',
                    id=email['id'],
                    body={
                        'removeLabelIds': ['INBOX'],
                        'addLabelIds': ['ARCHIVE']
                    }
                ).execute()
            
            # Remove from local clusters
            del self.clusters[cluster_id]
            if cluster_id in self.cluster_names:
                del self.cluster_names[cluster_id]
            
            return True
            
        except HttpError as error:
            return False

# Global assistant instance
assistant = InboxTriageAssistant()

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/load_emails')
def load_emails():
    """Load and cluster emails"""
    try:
        # Authenticate
        if not assistant.authenticate():
            return jsonify({'error': 'Authentication failed. Please check your credentials.'}), 401
        
        # Fetch emails
        if not assistant.fetch_emails():
            return jsonify({'error': 'Failed to fetch emails'}), 500
        
        # Cluster emails
        if not assistant.cluster_emails():
            return jsonify({'error': 'Failed to cluster emails'}), 500
        
        # Prepare data for frontend
        clusters_data = []
        for cluster_id, emails in assistant.clusters.items():
            cluster_name = assistant.cluster_names.get(cluster_id, f"Cluster {cluster_id}")
            
            # Format emails for display
            emails_data = []
            for email in emails[:5]:  # Show first 5 emails
                emails_data.append({
                    'subject': email['subject'][:60] + '...' if len(email['subject']) > 60 else email['subject'],
                    'sender': email['sender'],
                    'date': email['date'][:16] if email['date'] else 'Unknown date'
                })
            
            clusters_data.append({
                'id': int(cluster_id),  # Convert numpy.int32 to regular int
                'name': cluster_name,
                'email_count': int(len(emails)),  # Convert numpy.int32 to regular int
                'emails': emails_data
            })
        
        return jsonify({
            'success': True,
            'total_emails': int(len(assistant.emails)),  # Convert numpy.int32 to regular int
            'clusters': clusters_data
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/archive_cluster/<int:cluster_id>')
def archive_cluster(cluster_id):
    """Archive a specific cluster"""
    try:
        # Convert to int to handle any numpy types
        cluster_id = int(cluster_id)
        if assistant.archive_cluster(cluster_id):
            return jsonify({'success': True, 'message': f'Cluster {cluster_id} archived successfully'})
        else:
            return jsonify({'error': 'Failed to archive cluster'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/refresh')
def refresh():
    """Refresh emails and clusters"""
    try:
        # Clear existing data
        assistant.emails = []
        assistant.clusters = {}
        assistant.cluster_names = {}
        
        return jsonify({'success': True, 'message': 'Ready to load emails'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint for cloud deployment"""
    return jsonify({'status': 'healthy', 'message': 'Inbox Triage Assistant is running'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(debug=debug, host='0.0.0.0', port=port)
