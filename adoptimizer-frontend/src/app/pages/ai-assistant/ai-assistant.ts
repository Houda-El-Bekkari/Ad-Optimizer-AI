import { ChangeDetectorRef, Component } from '@angular/core';
import { FormsModule } from '@angular/forms';

import { AiAssistantApi, AiAssistantResponse, AssistantMode } from '../../services/ai-assistant-api';

interface ChatMessage {
  role: 'user' | 'ai';
  text: string;
  meta?: string;
}

@Component({
  selector: 'app-ai-assistant',
  imports: [FormsModule],
  templateUrl: './ai-assistant.html',
  styleUrl: './ai-assistant.scss',
})
export class AiAssistant {
  question = '';
  selectedMode: AssistantMode = 'auto';
  loading = false;
  error = '';

  readonly modes: Array<{ label: string; value: AssistantMode }> = [
    { label: 'Learning', value: 'learning' },
    { label: 'Decision', value: 'analysis' },
    { label: 'Auto', value: 'auto' },
  ];

  readonly promptGroups = [
    {
      title: 'Publicite digitale',
      prompts: ["Qu'est-ce que le ROAS ?", 'Comment fonctionne le CPA ?', 'Difference entre CPC et CPM ?'],
    },
    {
      title: 'Optimisation campagnes',
      prompts: ['Pourquoi mon CTR baisse ?', 'Comment reduire mon CPA ?', 'Quel budget reallouer ?'],
    },
    {
      title: 'Creation campagne',
      prompts: ['Quel budget recommandes-tu ?', 'Meta Ads ou Google Search ?', 'Comment prevoir le CPA ?'],
    },
  ];

  messages: ChatMessage[] = [
    {
      role: 'user',
      text: 'Pourquoi mon ROAS a baisse cette semaine ?',
      meta: 'Exemple',
    },
    {
      role: 'ai',
      text:
        'Je peux analyser les dernieres optimisations stockees et expliquer les causes probables, les risques et les actions prioritaires.',
      meta: 'AdOptimizer AI',
    },
  ];

  lastMode = 'auto';

  constructor(
    private readonly api: AiAssistantApi,
    private readonly cdr: ChangeDetectorRef,
  ) {}

  selectMode(mode: AssistantMode): void {
    this.selectedMode = mode;
  }

  usePrompt(prompt: string): void {
    this.question = prompt;
    this.ask();
  }

  ask(): void {
    const question = this.question.trim();

    if (!question || this.loading) {
      return;
    }

    this.error = '';
    this.loading = true;
    this.messages = [...this.messages, { role: 'user', text: question, meta: 'Vous' }];
    this.question = '';

    this.api.ask({ question, mode: this.selectedMode }).subscribe({
      next: (response) => {
        this.lastMode = String(response.mode || this.selectedMode);
        this.messages = [
          ...this.messages,
          {
            role: 'ai',
            text: this.extractAnswer(response),
            meta: this.modeLabel(this.lastMode),
          },
        ];
        this.loading = false;
        this.cdr.detectChanges();
      },
      error: () => {
        this.error =
          "Impossible de joindre le workflow chatbot. Verifiez que n8n est lance, que le workflow est publie et que FastAPI est actif.";
        this.messages = [
          ...this.messages,
          {
            role: 'ai',
            text: "Je n'arrive pas encore a joindre le workflow chatbot. Lancez n8n et FastAPI, puis reessayez.",
            meta: 'Connexion',
          },
        ];
        this.loading = false;
        this.cdr.detectChanges();
      },
    });
  }

  private extractAnswer(response: AiAssistantResponse): string {
    if (typeof response.answer === 'string' && response.answer.trim()) {
      return response.answer.trim();
    }

    const nested = response['data'];
    if (nested && typeof nested === 'object' && 'answer' in nested) {
      const answer = (nested as { answer?: unknown }).answer;
      if (typeof answer === 'string' && answer.trim()) {
        return answer.trim();
      }
    }

    for (const key of ['response', 'message', 'text', 'output']) {
      const value = response[key];
      if (typeof value === 'string' && value.trim()) {
        return value.trim();
      }
    }

    return 'Reponse recue, mais son format doit etre verifie dans le workflow n8n.';
  }

  private modeLabel(mode: string): string {
    if (mode === 'analysis') {
      return 'Decision';
    }

    if (mode === 'learning') {
      return 'Learning';
    }

    return 'Auto';
  }
}
