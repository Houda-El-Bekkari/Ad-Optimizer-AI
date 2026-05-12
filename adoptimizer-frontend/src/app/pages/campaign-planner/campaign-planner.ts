import { ChangeDetectorRef, Component } from '@angular/core';
import { FormsModule } from '@angular/forms';

import {
  CampaignPlannerApi,
  CampaignPlannerChannelPlan,
  CampaignPlannerRequest,
  CampaignPlannerResponse,
} from '../../services/campaign-planner-api';

@Component({
  selector: 'app-campaign-planner',
  imports: [FormsModule],
  templateUrl: './campaign-planner.html',
  styleUrls: ['../shared/internal-page.scss', './campaign-planner.scss'],
})
export class CampaignPlanner {
  form: CampaignPlannerRequest = {
    objectif: 'leads',
    budget: 3000,
    plateforme: 'both',
    produit: 'SaaS CRM B2B',
  };

  result?: CampaignPlannerResponse;
  loading = false;
  error = '';

  constructor(
    private readonly api: CampaignPlannerApi,
    private readonly changeDetector: ChangeDetectorRef,
  ) {}

  get recommendation() {
    return this.result?.recommendation;
  }

  get multiPlatformPlan() {
    return this.result?.multi_platform_plan;
  }

  get multiPlatformChannels(): CampaignPlannerChannelPlan[] {
    return this.multiPlatformPlan?.channels ?? [];
  }

  get hasMultiPlatformPlan(): boolean {
    return this.multiPlatformChannels.length > 0;
  }

  get recommendationTitle(): string {
    if (this.hasMultiPlatformPlan) {
      return 'Plan multi-plateforme recommande';
    }

    return this.recommendation?.name || 'Strategie IA en attente';
  }

  get recommendationPlatformLabel(): string {
    if (this.hasMultiPlatformPlan) {
      return 'Meta Ads + Google Ads';
    }

    return this.platformLabel(this.recommendation?.platform || this.form.plateforme);
  }

  get recommendationBudgetLabel(): string {
    if (this.multiPlatformPlan?.total_budget) {
      return this.displayMoney(this.multiPlatformPlan.total_budget);
    }

    return this.displayMoney(this.recommendation?.budget, this.displayMoney(this.form.budget));
  }

  get kpis() {
    return this.recommendation?.kpis;
  }

  get confidence() {
    return this.result?.confidence;
  }

  get whyList(): string[] {
    const multiWhy = Array.isArray(this.result?.multi_platform_why) ? this.result.multi_platform_why : [];
    const why = Array.isArray(this.result?.why) ? this.result.why : [];

    return multiWhy.length ? [...multiWhy, ...why] : why;
  }

  get risksList(): string[] {
    return Array.isArray(this.result?.risks) ? this.result.risks : [];
  }

  get actionPlanList() {
    const multiActionPlan = Array.isArray(this.result?.multi_platform_action_plan)
      ? this.result.multi_platform_action_plan
      : [];
    const actionPlan = Array.isArray(this.result?.action_plan) ? this.result.action_plan : [];

    return multiActionPlan.length ? multiActionPlan : actionPlan;
  }

  budgetShare(value: unknown): number {
    const total = Number(this.multiPlatformPlan?.total_budget || this.form.budget);
    const numberValue = Number(value);

    if (!Number.isFinite(total) || total <= 0 || !Number.isFinite(numberValue) || numberValue <= 0) {
      return 0;
    }

    return Math.max(4, Math.min(100, (numberValue / total) * 100));
  }

  generateStrategy(): void {
    this.loading = true;
    this.error = '';

    const payload: CampaignPlannerRequest = {
      ...this.form,
      budget: Number(this.form.budget) || 0,
    };

    this.api.generateStrategy(payload).subscribe({
      next: (result) => {
        this.result = result;
        this.loading = false;
        this.changeDetector.detectChanges();
      },
      error: (error: unknown) => {
        this.error = this.getErrorMessage(error);
        this.loading = false;
        this.changeDetector.detectChanges();
      },
    });
  }

  resetForm(): void {
    this.form = {
      objectif: 'leads',
      budget: 3000,
      plateforme: 'both',
      produit: 'SaaS CRM B2B',
    };
    this.result = undefined;
    this.error = '';
    this.changeDetector.detectChanges();
  }

  platformLabel(platform?: string): string {
    const labels: Record<string, string> = {
      both: 'Meta Ads + Google Ads',
      meta: 'Meta Ads',
      google: 'Google Ads',
    };

    return labels[platform ?? ''] ?? platform ?? 'Plateforme recommandee';
  }

  displayNumber(value: unknown, fallback = '--'): string {
    const numberValue = Number(value);
    return Number.isFinite(numberValue) ? Math.round(numberValue).toString() : fallback;
  }

  displayMoney(value: unknown, fallback = '--'): string {
    const numberValue = Number(value);
    return Number.isFinite(numberValue) ? `${numberValue.toFixed(2)} EUR` : fallback;
  }

  displayRoas(value: unknown, fallback = '--'): string {
    const numberValue = Number(value);
    return Number.isFinite(numberValue) ? `${numberValue.toFixed(3)}x` : fallback;
  }

  private getErrorMessage(error: unknown): string {
    const response = error as {
      error?: { message?: unknown } | string;
      message?: unknown;
      status?: unknown;
      statusText?: unknown;
    };

    if (typeof response.error === 'string') {
      return response.error;
    }

    if (response.error && typeof response.error === 'object' && typeof response.error.message === 'string') {
      return response.error.message;
    }

    if (typeof response.message === 'string') {
      return response.message;
    }

    return 'Impossible de joindre le workflow n8n. Verifiez que n8n est lance et que le webhook de test est actif.';
  }
}
