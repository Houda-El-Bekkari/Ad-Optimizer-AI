import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';

export type AssistantMode = 'auto' | 'learning' | 'analysis';

export interface AiAssistantRequest {
  question: string;
  mode?: AssistantMode;
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
  private readonly webhookUrl = 'http://localhost:5678/webhook/llm-chatbot';

  constructor(private readonly http: HttpClient) {}

  ask(payload: AiAssistantRequest): Observable<AiAssistantResponse> {
    return this.http.post<AiAssistantResponse>(this.webhookUrl, payload);
  }
}
