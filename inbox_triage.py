#!/usr/bin/env python3
"""
Inbox Triage Assistant
Clusters emails and enables one-click archiving
"""

import os
import pickle
import base64
import re
from typing import List, Dict, Tuple
from collections import defaultdict

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

class InboxTriageAssistant:
    def __init__(self):
        self.service = None
        self.emails = []
        self.clusters = {}
        
    def authenticate(self):
        """Authenticate with Gmail API using service account only"""
        try:
            # Try to get service account credentials from environment
            service_account_info = os.environ.get('GOOGLE_SERVICE_ACCOUNT')
            
            if service_account_info:
                print("Using service account authentication")
                # Parse the service account JSON from environment variable
                service_account_dict = json.loads(service_account_info)
                creds = service_account.Credentials.from_service_account_info(
                    service_account_dict, scopes=SCOPES
                )
                self.service = build('gmail', 'v1', credentials=creds)
                print("Gmail service built successfully with service account")
                return True
            else:
                print("No service account credentials found in GOOGLE_SERVICE_ACCOUNT environment variable")
                print("Please set GOOGLE_SERVICE_ACCOUNT environment variable with your service account JSON")
                return False
                
        except Exception as e:
            print(f"Authentication failed: {str(e)}")
            return False
    
    def fetch_emails(self, max_emails=200):
        """Fetch last 200 emails from inbox"""
        try:
            print("📧 Fetching emails...")
            
            # Get email IDs
            results = self.service.users().messages().list(
                userId='me', 
                labelIds=['INBOX'],
                maxResults=max_emails
            ).execute()
            
            messages = results.get('messages', [])
            if not messages:
                print("No emails found in inbox")
                return
            
            # Fetch full email details
            self.emails = []
            for i, message in enumerate(messages):
                print(f"Processing email {i+1}/{len(messages)}...")
                
                msg = self.service.users().messages().get(
                    userId='me', 
                    id=message['id'],
                    format='full'
                ).execute()
                
                email_data = self._parse_email(msg)
                if email_data:
                    self.emails.append(email_data)
            
            print(f"✅ Fetched {len(self.emails)} emails")
            
        except HttpError as error:
            print(f"❌ Error fetching emails: {error}")
    
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
            print(f"Error parsing email: {e}")
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
            print("No emails to cluster")
            return
        
        print("🔍 Clustering emails...")
        
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
        
        print(f"✅ Created {len(self.clusters)} clusters")
    
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
                names[cluster_id] = f"📧 {most_common.title()} ({len(emails)} emails)"
            else:
                # Fallback to sender domain
                domains = []
                for sender in senders:
                    domain = sender.split('@')[-1] if '@' in sender else sender
                    domains.append(domain)
                
                if domains:
                    most_common_domain = max(set(domains), key=domains.count)
                    names[cluster_id] = f"📧 {most_common_domain} ({len(emails)} emails)"
                else:
                    names[cluster_id] = f"📧 Cluster {cluster_id + 1} ({len(emails)} emails)"
        
        return names
    
    def display_clusters(self):
        """Display clusters with options"""
        print("\n" + "="*60)
        print("📬 INBOX TRIAGE ASSISTANT")
        print("="*60)
        
        for cluster_id, emails in self.clusters.items():
            cluster_name = self.cluster_names[cluster_id]
            print(f"\n{cluster_name}")
            print("-" * 40)
            
            # Show first 3 emails as examples
            for i, email in enumerate(emails[:3]):
                print(f"  {i+1}. {email['subject'][:50]}...")
                print(f"     From: {email['sender']}")
            
            if len(emails) > 3:
                print(f"  ... and {len(emails) - 3} more emails")
        
        print("\n" + "="*60)
    
    def archive_cluster(self, cluster_id):
        """Archive all emails in a cluster"""
        if cluster_id not in self.clusters:
            print("❌ Invalid cluster ID")
            return
        
        emails = self.clusters[cluster_id]
        print(f"🗄️ Archiving {len(emails)} emails...")
        
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
            
            print(f"✅ Successfully archived {len(emails)} emails!")
            
            # Remove from local clusters
            del self.clusters[cluster_id]
            
        except HttpError as error:
            print(f"❌ Error archiving emails: {error}")
    
    def run(self):
        """Main application loop"""
        print("🚀 Starting Inbox Triage Assistant...")
        
        # Authenticate
        if not self.authenticate():
            return
        
        # Fetch emails
        self.fetch_emails()
        
        if not self.emails:
            print("No emails found. Exiting.")
            return
        
        # Cluster emails
        self.cluster_emails()
        
        # Main interaction loop
        while True:
            self.display_clusters()
            
            if not self.clusters:
                print("All emails have been processed!")
                break
            
            print("\nOptions:")
            print("  Enter cluster number to archive (e.g., 0, 1, 2...)")
            print("  Enter 'q' to quit")
            print("  Enter 'r' to refresh")
            
            choice = input("\nYour choice: ").strip().lower()
            
            if choice == 'q':
                print("👋 Goodbye!")
                break
            elif choice == 'r':
                print("🔄 Refreshing...")
                self.fetch_emails()
                self.cluster_emails()
            elif choice.isdigit():
                cluster_id = int(choice)
                if cluster_id in self.clusters:
                    self.archive_cluster(cluster_id)
                else:
                    print("❌ Invalid cluster number")
            else:
                print("❌ Invalid choice")

if __name__ == "__main__":
    assistant = InboxTriageAssistant()
    assistant.run()
