import { ChangeDetectorRef, Component, OnInit } from '@angular/core';
import { FormsModule } from '@angular/forms';

import { AiAssistantApi, AiAssistantResponse, AssistantMode, ChatHistoryItem } from '../../services/ai-assistant-api';
import { AuthApi } from '../../services/auth-api';

interface ChatMessage {
  id?: number;
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
export class AiAssistant implements OnInit {
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

  private readonly initialMessages: ChatMessage[] = [
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

  messages: ChatMessage[] = [...this.initialMessages];
  lastMode = 'auto';
  deletingMessageId: number | null = null;
  clearingHistory = false;

  constructor(
    private readonly api: AiAssistantApi,
    private readonly authApi: AuthApi,
    private readonly cdr: ChangeDetectorRef,
  ) {}

  ngOnInit(): void {
    this.loadHistory();
  }

  get hasSavedMessages(): boolean {
    return this.messages.some((message) => typeof message.id === 'number');
  }

  loadHistory(): void {
    if (!this.authApi.isAuthenticated()) {
      return;
    }

    this.api.history().subscribe({
      next: (history) => {
        this.applyHistory(history);
        this.cdr.detectChanges();
      },
      error: (error) => {
        console.error('Unable to load chatbot history', error);
      },
    });
  }

  deleteMessage(message: ChatMessage): void {
    if (!message.id || !this.authApi.isAuthenticated() || this.deletingMessageId) {
      return;
    }

    this.deletingMessageId = message.id;

    this.api.deleteMessage(message.id).subscribe({
      next: () => {
        this.deletingMessageId = null;
        this.loadHistory();
      },
      error: (error) => {
        console.error('Unable to delete chatbot message', error);
        this.deletingMessageId = null;
        this.cdr.detectChanges();
      },
    });
  }

  clearConversation(): void {
    if (!this.authApi.isAuthenticated() || this.clearingHistory || !this.hasSavedMessages) {
      return;
    }

    if (!window.confirm('Supprimer toute la conversation ?')) {
      return;
    }

    this.clearingHistory = true;

    this.api.clearHistory().subscribe({
      next: () => {
        this.clearingHistory = false;
        this.loadHistory();
      },
      error: (error) => {
        console.error('Unable to clear chatbot history', error);
        this.clearingHistory = false;
        this.cdr.detectChanges();
      },
    });
  }

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

    if (!this.authApi.isAuthenticated()) {
      this.error = 'Connectez-vous pour utiliser la memoire conversationnelle du chatbot.';
      return;
    }

    this.error = '';
    this.loading = true;
    this.messages = [...this.messages, { role: 'user', text: question, meta: 'Vous' }];
    this.question = '';

    this.api.ask({ question, mode: this.selectedMode }).subscribe({
      next: (response) => {
        this.lastMode = String(response.mode || this.selectedMode);
        const answer = this.extractAnswer(response);
        this.messages = [
          ...this.messages,
          {
            role: 'ai',
            text: answer,
            meta: this.modeLabel(this.lastMode),
          },
        ];
        this.loading = false;
        this.loadHistory();
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

  private applyHistory(history: ChatHistoryItem[]): void {
    if (!history.length) {
      this.messages = [...this.initialMessages];
      return;
    }

    this.messages = history.flatMap((item) => [
      {
        id: item.id,
        role: 'user' as const,
        text: item.question,
        meta: 'Vous',
      },
      {
        id: item.id,
        role: 'ai' as const,
        text: item.response,
        meta: this.modeLabel(item.mode),
      },
    ]);
    this.lastMode = history[history.length - 1].mode || this.lastMode;
  }

}
