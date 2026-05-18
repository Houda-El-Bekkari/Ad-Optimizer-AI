import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';

import { environment } from '../../environments/environment';

export type AssistantMode = 'auto' | 'learning' | 'analysis';

export interface AiAssistantRequest {
  question: string;
  mode?: AssistantMode;
}

export interface ChatHistoryItem {
  id: number;
  user_id: number;
  question: string;
  response: string;
  mode: string;
  created_at: string;
}

export interface SaveChatMessageRequest {
  question: string;
  response: string;
  mode?: AssistantMode | string;
}

export interface DeleteChatResponse {
  success: boolean;
  message: string;
  deleted_count?: number;
}

export interface AiAssistantResponse {
  answer?: string;
  mode?: string;
  status?: string;
  domain?: string[];
  sources?: string[];
  [key: string]: unknown;
}

@Injectable({ providedIn: 'root' })
export class AiAssistantApi {
  private readonly apiUrl = environment.apiUrl;

  constructor(private readonly http: HttpClient) {}

  ask(payload: AiAssistantRequest): Observable<AiAssistantResponse> {
    return this.http.post<AiAssistantResponse>(`${this.apiUrl}/chatbot/ask`, payload);
  }

  history(): Observable<ChatHistoryItem[]> {
    return this.http.get<ChatHistoryItem[]>(`${this.apiUrl}/chatbot/history`);
  }

  saveMessage(payload: SaveChatMessageRequest): Observable<ChatHistoryItem> {
    return this.http.post<ChatHistoryItem>(`${this.apiUrl}/chatbot/messages`, payload);
  }

  deleteMessage(messageId: number): Observable<DeleteChatResponse> {
    return this.http.delete<DeleteChatResponse>(`${this.apiUrl}/chatbot/messages/${messageId}`);
  }

  clearHistory(): Observable<DeleteChatResponse> {
    return this.http.delete<DeleteChatResponse>(`${this.apiUrl}/chatbot/history`);
  }
}
