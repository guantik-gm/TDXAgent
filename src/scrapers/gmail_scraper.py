"""
Gmail scraper for TDXAgent.

This module provides Gmail integration using the official Gmail API
with OAuth 2.0 authentication for secure email data collection.
"""

import asyncio
import base64
import email
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from email.mime.text import MIMEText
from email.utils import parsedate_to_datetime

try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
except ImportError:
    print("Gmail dependencies not installed. Please run: pip install google-auth google-auth-oauthlib google-api-python-client")
    raise

from .base_scraper import BaseScraper, ScrapingResult
from utils.logger import TDXLogger


class GmailScraper(BaseScraper):
    """
    Gmail scraper using official Gmail API.
    
    Features:
    - OAuth 2.0 authentication
    - Time-based email filtering
    - Smart filtering (labels, keywords, senders)
    - Message threading support
    - Rate limiting compliance
    """
    
    # Gmail API scopes - read-only access
    SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Gmail scraper with configuration."""
        super().__init__(config, 'gmail')
        
        # Gmail API configuration
        self.credentials_file = config.get('credentials_file', 'credentials.json')
        self.token_file = config.get('token_file', 'gmail_token.json')
        
        # Filtering configuration
        self.filters = config.get('filters', {})
        self.labels = self.filters.get('labels', [])
        self.from_addresses = self.filters.get('from_addresses', [])
        self.keywords = self.filters.get('keywords', [])
        self.exclude_keywords = self.filters.get('exclude_keywords', [])
        self.exclude_spam = self.filters.get('exclude_spam', True)
        
        # API configuration
        self.batch_size = config.get('batch_size', 100)
        self.max_results = config.get('max_results', 500)
        
        # Gmail API service
        self.service = None
        self.credentials = None
        
        self.logger.info("Gmail scraper initialized")
    
    def _setup_proxy(self) -> None:
        """Setup proxy configuration for Gmail API requests."""
        try:
            # Get proxy configuration from config manager
            from config.config_manager import ConfigManager
            config_manager = ConfigManager()
            
            if config_manager.proxy.enabled:
                import os
                proxy_url = None
                
                if config_manager.proxy.type in ['http', 'https']:
                    # HTTP/HTTPS proxy
                    if config_manager.proxy.username and config_manager.proxy.password:
                        proxy_url = f"http://{config_manager.proxy.username}:{config_manager.proxy.password}@{config_manager.proxy.host}:{config_manager.proxy.port}"
                    else:
                        proxy_url = f"http://{config_manager.proxy.host}:{config_manager.proxy.port}"
                    
                    self.logger.info(f"Using HTTP proxy: {config_manager.proxy.host}:{config_manager.proxy.port}")
                    
                elif config_manager.proxy.type == 'socks5':
                    # SOCKS5 proxy - convert to HTTP proxy format for Google API
                    if config_manager.proxy.username and config_manager.proxy.password:
                        proxy_url = f"socks5://{config_manager.proxy.username}:{config_manager.proxy.password}@{config_manager.proxy.host}:{config_manager.proxy.port}"
                    else:
                        proxy_url = f"socks5://{config_manager.proxy.host}:{config_manager.proxy.port}"
                    
                    self.logger.info(f"Using SOCKS5 proxy: {config_manager.proxy.host}:{config_manager.proxy.port}")
                
                if proxy_url:
                    # Set proxy environment variables for Google API client
                    os.environ['HTTP_PROXY'] = proxy_url
                    os.environ['HTTPS_PROXY'] = proxy_url
                    
                    # Also set lowercase versions (some libraries use these)
                    os.environ['http_proxy'] = proxy_url
                    os.environ['https_proxy'] = proxy_url
                    
                    self.logger.info("Proxy environment variables set for Gmail API")
            else:
                # Clear proxy environment variables if they exist
                import os
                proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']
                for var in proxy_vars:
                    if var in os.environ:
                        del os.environ[var]
                
        except Exception as e:
            self.logger.warning(f"Failed to setup proxy configuration: {e}")
    
    async def authenticate(self) -> bool:
        """
        Authenticate with Gmail using OAuth 2.0.
        
        Returns:
            True if authentication successful, False otherwise
        """
        try:
            self.logger.info("Starting Gmail OAuth authentication...")
            
            # Setup proxy configuration if enabled
            self._setup_proxy()
            
            # Load existing credentials
            if self._load_existing_credentials():
                self.logger.info("Loaded existing Gmail credentials")
            else:
                # Run OAuth flow
                if not await self._run_oauth_flow():
                    return False
            
            # Build Gmail service
            self.service = build('gmail', 'v1', credentials=self.credentials)
            
            # Test authentication with timeout
            import socket
            socket.setdefaulttimeout(30)  # 30 second timeout
            
            profile = self.service.users().getProfile(userId='me').execute()
            email_address = profile.get('emailAddress')
            
            self.logger.info(f"Gmail authentication successful for: {email_address}")
            self._is_authenticated = True
            return True
            
        except Exception as e:
            self.logger.error(f"Gmail authentication failed: {e}")
            return False
    
    def _load_existing_credentials(self) -> bool:
        """Load existing credentials from token file."""
        try:
            import os
            if os.path.exists(self.token_file):
                self.credentials = Credentials.from_authorized_user_file(
                    self.token_file, self.SCOPES
                )
                
                # Refresh if expired
                if self.credentials and self.credentials.expired and self.credentials.refresh_token:
                    try:
                        self.logger.info("Refreshing expired Gmail credentials...")
                        # Ensure proxy is set for credential refresh
                        self._setup_proxy()
                        self.credentials.refresh(Request())
                        # Save refreshed credentials
                        with open(self.token_file, 'w') as token:
                            token.write(self.credentials.to_json())
                        self.logger.info("Gmail credentials refreshed successfully")
                    except Exception as refresh_error:
                        self.logger.error(f"Failed to refresh Gmail credentials: {refresh_error}")
                        self.logger.error(f"This may be due to network connectivity or proxy configuration issues")
                        return False
                
                return self.credentials and self.credentials.valid
            return False
        except Exception as e:
            self.logger.warning(f"Failed to load existing credentials: {e}")
            return False
    
    async def _run_oauth_flow(self) -> bool:
        """Run OAuth 2.0 flow for new authentication."""
        try:
            import os
            if not os.path.exists(self.credentials_file):
                self.logger.error(
                    f"Gmail credentials file not found: {self.credentials_file}\n"
                    "Please download credentials.json from Google Cloud Console:\n"
                    "1. Visit https://console.cloud.google.com/\n"
                    "2. Enable Gmail API\n"
                    "3. Create OAuth 2.0 credentials\n"
                    "4. Download credentials.json"
                )
                return False
            
            # Ensure proxy is set for OAuth flow
            self._setup_proxy()
            
            # Run OAuth flow
            flow = InstalledAppFlow.from_client_secrets_file(
                self.credentials_file, self.SCOPES
            )
            
            self.logger.info("Opening browser for Gmail OAuth authentication...")
            self.credentials = flow.run_local_server(port=0)
            
            # Save credentials
            with open(self.token_file, 'w') as token:
                token.write(self.credentials.to_json())
            
            self.logger.info("Gmail OAuth authentication completed")
            return True
            
        except Exception as e:
            self.logger.error(f"OAuth flow failed: {e}")
            self.logger.error(f"This may be due to network connectivity or proxy configuration issues")
            return False
    
    async def scrape(self, hours_back: int = 12, **kwargs) -> ScrapingResult:
        """
        Scrape emails from Gmail.
        
        Args:
            hours_back: Number of hours back to scrape
            **kwargs: Additional parameters
            
        Returns:
            ScrapingResult containing email data
        """
        if not self.service:
            raise Exception("Gmail service not initialized. Call authenticate() first.")
        
        start_time = datetime.now()
        self._session_start_time = start_time
        messages = []
        errors = []
        
        try:
            self.logger.info(f"Starting Gmail scraping for last {hours_back} hours")
            
            # Build search query
            query = self._build_search_query(hours_back)
            self.logger.info(f"Gmail search query: {query}")
            
            # Get message list
            message_ids = await self._get_message_list(query)
            self.logger.info(f"Found {len(message_ids)} emails matching criteria")
            
            # Get message details in batches
            messages = await self._get_messages_batch(message_ids)
            
            # Filter and validate messages
            valid_messages = []
            for msg in messages:
                if self.validate_message(msg):
                    valid_messages.append(msg)
                else:
                    errors.append(f"Invalid message format: {msg.get('id', 'unknown')}")
            
            self.logger.info(
                f"Gmail scraping completed: {len(valid_messages)}/{len(messages)} "
                f"valid messages"
            )
            
            return self.create_scraping_result(valid_messages, errors)
            
        except Exception as e:
            error_msg = f"Gmail scraping failed: {e}"
            self.logger.error(error_msg)
            errors.append(error_msg)
            return self.create_scraping_result(messages, errors)
    
    def _build_search_query(self, hours_back: int) -> str:
        """Build Gmail search query with filters."""
        query_parts = []
        
        # Time filter
        after_date = datetime.now() - timedelta(hours=hours_back)
        query_parts.append(f"after:{after_date.strftime('%Y/%m/%d')}")
        
        # Label filters
        for label in self.labels:
            query_parts.append(f"label:{label}")
        
        # From address filters
        for address in self.from_addresses:
            query_parts.append(f"from:{address}")
        
        # Keyword filters (include)
        for keyword in self.keywords:
            query_parts.append(f'"{keyword}"')
        
        # Exclude keyword filters
        for exclude_keyword in self.exclude_keywords:
            query_parts.append(f'-"{exclude_keyword}"')
        
        # Exclude spam
        if self.exclude_spam:
            query_parts.append("-in:spam")
            query_parts.append("-in:trash")
        
        return " ".join(query_parts) if query_parts else "in:inbox"
    
    async def _get_message_list(self, query: str) -> List[str]:
        """Get list of message IDs matching query."""
        try:
            message_ids = []
            next_page_token = None
            
            while len(message_ids) < self.max_results:
                # Apply rate limiting
                await self.rate_limit()
                
                # Get message list page
                request = self.service.users().messages().list(
                    userId='me',
                    q=query,
                    maxResults=min(self.batch_size, self.max_results - len(message_ids)),
                    pageToken=next_page_token
                )
                
                result = request.execute()
                messages = result.get('messages', [])
                
                if not messages:
                    break
                
                # Extract message IDs
                page_ids = [msg['id'] for msg in messages]
                message_ids.extend(page_ids)
                
                # Check for next page
                next_page_token = result.get('nextPageToken')
                if not next_page_token:
                    break
                
                self.logger.debug(f"Retrieved {len(page_ids)} message IDs (total: {len(message_ids)})")
            
            return message_ids[:self.max_results]
            
        except HttpError as e:
            self.logger.error(f"Failed to get message list: {e}")
            return []
    
    async def _get_messages_batch(self, message_ids: List[str]) -> List[Dict[str, Any]]:
        """Get detailed message data in batches."""
        messages = []
        
        # Process in batches to avoid rate limits
        batch_size = 50  # Gmail API batch limit
        for i in range(0, len(message_ids), batch_size):
            batch_ids = message_ids[i:i + batch_size]
            batch_messages = await self._get_message_batch(batch_ids)
            messages.extend(batch_messages)
            
            self.logger.debug(f"Processed batch {i//batch_size + 1}/{(len(message_ids) + batch_size - 1)//batch_size}")
        
        return messages
    
    async def _get_message_batch(self, message_ids: List[str]) -> List[Dict[str, Any]]:
        """Get a batch of message details."""
        messages = []
        
        for msg_id in message_ids:
            try:
                # Apply rate limiting
                await self.rate_limit()
                
                # Get message details
                message = self.service.users().messages().get(
                    userId='me',
                    id=msg_id,
                    format='full'
                ).execute()
                
                # Convert to standard format
                formatted_msg = self._format_gmail_message(message)
                if formatted_msg:
                    messages.append(formatted_msg)
                
            except Exception as e:
                self.logger.warning(f"Failed to get message {msg_id}: {e}")
                continue
        
        return messages
    
    def _format_gmail_message(self, gmail_msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert Gmail message to standard TDXAgent format."""
        try:
            msg_id = gmail_msg['id']
            thread_id = gmail_msg['threadId']
            
            # Extract headers
            headers = {h['name']: h['value'] for h in gmail_msg['payload'].get('headers', [])}
            
            # Extract basic info
            from_header = headers.get('From', '')
            subject = headers.get('Subject', '')
            date_header = headers.get('Date', '')
            
            # Parse from header
            author_name, author_email = self._parse_from_header(from_header)
            
            # Parse date
            posted_at = self._parse_date_header(date_header)
            
            # Extract message body
            body_text = self._extract_message_body(gmail_msg['payload'])
            
            # Create message URL for Gmail web interface
            message_url = f"https://mail.google.com/mail/u/0/#inbox/{msg_id}"
            
            # Format in standard TDXAgent format
            formatted_message = {
                'id': f"tdx_gmail_{msg_id}",
                'platform': 'gmail',
                'author': {
                    'name': author_name,
                    'id': author_email,
                    'username': author_email
                },
                'content': {
                    'text': f"{subject}\n\n{body_text}".strip(),
                    'media': []  # No media support as requested
                },
                'metadata': {
                    'posted_at': posted_at,
                    'message_url': message_url,
                    'platform_specific': {
                        'thread_id': thread_id,
                        'subject': subject,
                        'labels': gmail_msg.get('labelIds', [])
                    }
                },
                'context': {
                    'channel': 'inbox',  # Default to inbox
                    'group': '',
                    'server': '',
                    'thread': thread_id
                }
            }
            
            return formatted_message
            
        except Exception as e:
            self.logger.warning(f"Failed to format Gmail message: {e}")
            return None
    
    def _parse_from_header(self, from_header: str) -> tuple[str, str]:
        """Parse From header to extract name and email."""
        try:
            # Handle formats like "Name <email@domain.com>" or just "email@domain.com"
            if '<' in from_header and '>' in from_header:
                # Extract name and email
                match = re.match(r'^(.*?)\s*<(.+?)>$', from_header.strip())
                if match:
                    name = match.group(1).strip(' "')
                    email = match.group(2).strip()
                    return name or email, email
            
            # Just email address
            email = from_header.strip()
            return email, email
            
        except Exception:
            return from_header, from_header
    
    def _parse_date_header(self, date_header: str) -> str:
        """Parse email date header to ISO format."""
        try:
            if date_header:
                dt = parsedate_to_datetime(date_header)
                return dt.isoformat()
            return datetime.now().isoformat()
        except Exception:
            return datetime.now().isoformat()
    
    def _extract_message_body(self, payload: Dict[str, Any]) -> str:
        """Extract text content from email payload."""
        try:
            body_text = ""
            
            # Handle multipart messages
            if payload.get('parts'):
                for part in payload['parts']:
                    part_body = self._extract_part_body(part)
                    if part_body:
                        body_text += part_body + "\n"
            else:
                # Single part message
                body_text = self._extract_part_body(payload)
            
            return body_text.strip()
            
        except Exception as e:
            self.logger.warning(f"Failed to extract message body: {e}")
            return ""
    
    def _extract_part_body(self, part: Dict[str, Any]) -> str:
        """Extract body from a single message part."""
        try:
            mime_type = part.get('mimeType', '')
            
            # Only process text parts
            if not mime_type.startswith('text/'):
                return ""
            
            # Get body data
            body = part.get('body', {})
            data = body.get('data', '')
            
            if not data:
                return ""
            
            # Decode base64
            decoded_bytes = base64.urlsafe_b64decode(data + '===')  # Add padding
            text = decoded_bytes.decode('utf-8', errors='ignore')
            
            # Clean up HTML if needed
            if mime_type == 'text/html':
                text = self._clean_html(text)
            
            return text
            
        except Exception as e:
            self.logger.warning(f"Failed to extract part body: {e}")
            return ""
    
    def _clean_html(self, html_text: str) -> str:
        """Basic HTML cleaning to extract text content."""
        try:
            # Remove HTML tags
            import re
            text = re.sub(r'<[^>]+>', '', html_text)
            
            # Decode HTML entities
            import html
            text = html.unescape(text)
            
            # Clean up whitespace
            text = re.sub(r'\s+', ' ', text)
            
            return text.strip()
            
        except Exception:
            return html_text
    
    async def cleanup(self) -> None:
        """Clean up Gmail API resources."""
        try:
            if self.service:
                # Gmail API doesn't need explicit cleanup
                self.service = None
            self.logger.info("Gmail scraper cleanup completed")
        except Exception as e:
            self.logger.warning(f"Gmail cleanup failed: {e}")
    
    def __str__(self) -> str:
        return f"GmailScraper(authenticated={self._is_authenticated})"