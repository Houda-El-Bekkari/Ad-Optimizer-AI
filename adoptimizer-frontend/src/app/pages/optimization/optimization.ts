import { ChangeDetectorRef, Component, OnInit } from '@angular/core';

import {
  CampaignOptimizationApi,
  CampaignOptimizationRecord,
} from '../../services/campaign-optimization-api';

type JsonObject = Record<string, unknown>;

@Component({
  selector: 'app-optimization',
  templateUrl: './optimization.html',
  styleUrls: ['../shared/internal-page.scss', './optimization.scss'],
})
export class Optimization implements OnInit {
  records: CampaignOptimizationRecord[] = [];
  selectedKey = '';
  loading = true;
  error = '';

  constructor(
    private readonly api: CampaignOptimizationApi,
    private readonly cdr: ChangeDetectorRef,
  ) {}

  ngOnInit(): void {
    this.loadDashboard();
  }

  loadDashboard(): void {
    this.loading = true;
    this.error = '';

    this.api.getOptimizationDashboard().subscribe({
      next: (response) => {
        this.records = response.items ?? [];
        this.selectedKey = this.resolveSelectedKey(this.records);
        this.loading = false;
        this.cdr.detectChanges();
      },
      error: () => {
        this.records = [];
        this.loading = false;
        this.error =
          "Impossible de charger les resultats. Verifiez que le workflow dashboard n8n est publie.";
        this.cdr.detectChanges();
      },
    });
  }

  get primary(): CampaignOptimizationRecord | undefined {
    return this.records.find((record) => this.recordKey(record) === this.selectedKey) || this.records[0];
  }

  get currentKpis(): JsonObject {
    return this.asObject(this.primary?.current_kpis);
  }

  get predictedKpis(): JsonObject {
    return this.asObject(this.primary?.predicted_kpis);
  }

  get expectedImpact(): JsonObject {
    return this.asObject(this.primary?.expected_impact);
  }

  get trend(): JsonObject {
    return this.asObject(this.primary?.trend);
  }

  get anomalyTypes(): string[] {
    return this.asItems<string>(this.primary?.anomaly_types);
  }

  get topAnomalies(): JsonObject[] {
    return this.asItems<JsonObject>(this.primary?.top_anomalies);
  }

  get campaignLabel(): string {
    const record = this.primary;

    if (!record) {
      return 'Audit en attente';
    }

    if (this.records.length > 1) {
      return `${record.global_campaign_id || 'Campagne'} - ${this.records.length} canaux`;
    }

    return `${this.platformLabel(record.platform)} - ${record.campaign_id}`;
  }

  get selectedChannelLabel(): string {
    const record = this.primary;

    if (!record) {
      return 'Canal en attente';
    }

    return `${this.platformLabel(record.platform)} - ${record.campaign_id}`;
  }

  get healthStatus(): string {
    return this.primary?.health_status || 'UNKNOWN';
  }

  get actionTitle(): string {
    if (this.isMaintainAction) {
      const expectedRoas = this.toNumber(this.expectedImpact['expected_roas']);
      const deltaConversions = this.toNumber(this.expectedImpact['delta_conversions']);

      if ((expectedRoas !== undefined && expectedRoas < 1) || (deltaConversions !== undefined && deltaConversions < 0)) {
        return 'Maintenir sous surveillance';
      }

      return 'Maintenir le budget';
    }

    return this.primary?.action_label || this.primary?.recommended_action || 'Action en attente';
  }

  get businessAdvice(): string {
    const expectedRoas = this.toNumber(this.expectedImpact['expected_roas']);
    const deltaConversions = this.toNumber(this.expectedImpact['delta_conversions']);
    const action = String(this.primary?.recommended_action || '').toLowerCase();

    if (action.includes('continue_monitoring') || action.includes('monitoring')) {
      return "Aucun changement budgetaire requis : continuer le monitoring et verifier les KPIs au prochain audit.";
    }

    if (this.isMaintainAction) {
      if (expectedRoas !== undefined && expectedRoas < 1) {
        return "Ne pas scaler maintenant : surveiller 48h et revoir le ciblage, les creatives ou l'offre avant toute hausse de budget.";
      }

      if (deltaConversions !== undefined && deltaConversions < 0) {
        return "Maintenir le budget, mais suivre les conversions de pres avant d'augmenter l'investissement.";
      }

      return "Continuer la surveillance avant toute modification budgetaire.";
    }

    if (action.includes('decrease')) {
      return "Reduire progressivement et verifier que le CPA baisse sans casser le volume de conversions.";
    }

    if (action.includes('increase')) {
      return "Augmenter par palier et controler ROAS, CPA et conversions avant de scaler davantage.";
    }

    if (action.includes('reallocate')) {
      return "Reallouer en test controle, puis comparer les performances des canaux avant de generaliser.";
    }

    if (action.includes('pause')) {
      return "Mettre en pause seulement apres validation humaine, puis relancer avec un ciblage ou une creation corrigee.";
    }

    return "Appliquer l'action progressivement et suivre les KPIs prioritaires.";
  }

  get hasCorrectiveAction(): boolean {
    const action = String(this.primary?.recommended_action || '').toLowerCase();

    return Boolean(action) && !action.includes('continue_monitoring') && !action.includes('monitoring');
  }

  selectRecord(record: CampaignOptimizationRecord): void {
    this.selectedKey = this.recordKey(record);
  }

  recordKey(record: CampaignOptimizationRecord): string {
    return `${record.campaign_id}|${record.platform}`;
  }

  platformLabel(platform?: string): string {
    if (platform === 'meta') {
      return 'Meta Ads';
    }

    if (platform === 'google') {
      return 'Google Ads';
    }

    return platform || 'Plateforme';
  }

  displayNumber(value: unknown, fallback = '--'): string {
    const numberValue = Number(value);

    if (!Number.isFinite(numberValue)) {
      return fallback;
    }

    return numberValue.toFixed(numberValue >= 10 ? 1 : 2).replace(/\.0$/, '');
  }

  displayPercent(value: unknown, fallback = '--'): string {
    const numberValue = Number(value);

    if (!Number.isFinite(numberValue)) {
      return fallback;
    }

    return `${numberValue.toFixed(1)}%`;
  }

  displayRoas(value: unknown, fallback = '--'): string {
    const numberValue = Number(value);

    if (!Number.isFinite(numberValue)) {
      return fallback;
    }

    return `${numberValue.toFixed(2)}x`;
  }

  private get isMaintainAction(): boolean {
    const action = String(this.primary?.recommended_action || '').toLowerCase();
    const label = String(this.primary?.action_label || '').toLowerCase();

    return action.includes('maintain') || label.startsWith('maint');
  }

  anomalyTitle(anomaly: JsonObject): string {
    const raw = String(anomaly['explanation'] || 'Anomalie detectee');
    return raw.split(' | ')[0].replace(/^\[[^\]]+\]\s*/, '');
  }

  anomalyMeta(anomaly: JsonObject): string {
    const date = anomaly['date'] ? String(anomaly['date']) : 'Date recente';
    const score = this.displayNumber(anomaly['score']);
    const roas = this.displayRoas(anomaly['roas']);

    return `${date} - score ${score} - ROAS ${roas}`;
  }

  private asObject(value: unknown): JsonObject {
    if (!value) {
      return {};
    }

    if (typeof value === 'string') {
      try {
        return JSON.parse(value) as JsonObject;
      } catch {
        return {};
      }
    }

    if (typeof value === 'object' && !Array.isArray(value)) {
      return value as JsonObject;
    }

    return {};
  }

  private asItems<T>(value: unknown): T[] {
    if (!value) {
      return [];
    }

    if (Array.isArray(value)) {
      return value as T[];
    }

    const objectValue = this.asObject(value);
    const items = objectValue['items'];

    return Array.isArray(items) ? (items as T[]) : [];
  }

  private toNumber(value: unknown): number | undefined {
    const numberValue = Number(value);
    return Number.isFinite(numberValue) ? numberValue : undefined;
  }

  private resolveSelectedKey(records: CampaignOptimizationRecord[]): string {
    if (!records.length) {
      return '';
    }

    const currentStillExists = records.some((record) => this.recordKey(record) === this.selectedKey);

    if (currentStillExists) {
      return this.selectedKey;
    }

    const mostCritical = [...records].sort(
      (left, right) => Number(left.health_score ?? 100) - Number(right.health_score ?? 100),
    )[0];

    return this.recordKey(mostCritical);
  }
}
