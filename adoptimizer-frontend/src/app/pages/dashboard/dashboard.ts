import { ChangeDetectorRef, Component, OnInit } from '@angular/core';
import { RouterLink } from '@angular/router';

import {
  CampaignOptimizationApi,
  CampaignOptimizationRecord,
} from '../../services/campaign-optimization-api';

type JsonObject = Record<string, unknown>;

interface OverviewAnomaly {
  title: string;
  meta: string;
  level: string;
}

@Component({
  selector: 'app-dashboard',
  imports: [RouterLink],
  templateUrl: './dashboard.html',
  styleUrl: './dashboard.scss',
})
export class Dashboard implements OnInit {
  records: CampaignOptimizationRecord[] = [];
  generatedAt = '';
  loading = true;
  error = '';

  constructor(
    private readonly api: CampaignOptimizationApi,
    private readonly cdr: ChangeDetectorRef,
  ) {}

  ngOnInit(): void {
    this.loadOverview();
  }

  loadOverview(): void {
    this.loading = true;
    this.error = '';

    this.api.getOptimizationDashboard().subscribe({
      next: (response) => {
        this.records = response.items ?? [];
        this.generatedAt = response.generated_at || '';
        this.loading = false;
        this.cdr.detectChanges();
      },
      error: () => {
        this.records = [];
        this.loading = false;
        this.error =
          "Impossible de charger l'overview. Verifiez que le workflow dashboard n8n est publie.";
        this.cdr.detectChanges();
      },
    });
  }

  get generatedAtLabel(): string {
    if (!this.generatedAt) {
      return 'Audit en attente';
    }

    return `Last audit: ${new Date(this.generatedAt).toLocaleString('fr-FR')}`;
  }

  get platformsLabel(): string {
    const platforms = [...new Set(this.records.map((record) => this.platformLabel(record.platform)))];
    return platforms.length ? platforms.join(' + ') : 'Meta + Google';
  }

  get primary(): CampaignOptimizationRecord | undefined {
    return [...this.records].sort(
      (left, right) => Number(left.health_score ?? 100) - Number(right.health_score ?? 100),
    )[0];
  }

  get primaryCurrentKpis(): JsonObject {
    return this.asObject(this.primary?.current_kpis);
  }

  get primaryPredictedKpis(): JsonObject {
    return this.asObject(this.primary?.predicted_kpis);
  }

  get primaryExpectedImpact(): JsonObject {
    return this.asObject(this.primary?.expected_impact);
  }

  get globalRoas(): number | undefined {
    return this.averageFromRecords((record) => this.toNumber(this.asObject(record.current_kpis)['roas']));
  }

  get predictedRoas(): number | undefined {
    return this.averageFromRecords((record) => this.toNumber(this.asObject(record.predicted_kpis)['roas_h14']));
  }

  get totalSpend(): number {
    return this.sumFromRecords((record) => this.toNumber(this.asObject(record.current_kpis)['spend']));
  }

  get totalConversions(): number {
    return this.sumFromRecords((record) => this.toNumber(this.asObject(record.current_kpis)['conversions']));
  }

  get averageCpa(): number | undefined {
    const conversions = this.totalConversions;

    if (conversions <= 0) {
      return undefined;
    }

    return this.totalSpend / conversions;
  }

  get averageHealthScore(): number | undefined {
    return this.averageFromRecords((record) => this.toNumber(record.health_score));
  }

  get aiAlerts(): number {
    return this.records.filter((record) => {
      const status = String(record.health_status || record.anomaly_level || '').toLowerCase();
      return status.includes('critical') || status.includes('warning');
    }).length;
  }

  get criticalCampaigns(): number {
    return this.records.filter((record) =>
      String(record.health_status || record.anomaly_level || '').toLowerCase().includes('critical'),
    ).length;
  }

  get performanceBars(): number[] {
    const start = this.globalRoas ?? 0;
    const end = this.predictedRoas ?? start;
    const max = Math.max(start, end, 1);

    return Array.from({ length: 14 }, (_, index) => {
      const ratio = index / 13;
      const value = start + (end - start) * ratio;
      return Math.max(18, Math.min(88, 22 + (value / max) * 58));
    });
  }

  get topAnomalies(): OverviewAnomaly[] {
    const anomalies: OverviewAnomaly[] = [];

    for (const record of this.records) {
      const topItems = this.asItems<JsonObject>(record.top_anomalies);

      for (const anomaly of topItems.slice(0, 2)) {
        anomalies.push({
          title: this.anomalyTitle(anomaly),
          meta: `${this.platformLabel(record.platform)} - ${record.campaign_id} | ${this.anomalyMeta(anomaly)}`,
          level: record.anomaly_level || record.health_status || 'Alert',
        });
      }
    }

    return anomalies.slice(0, 4);
  }

  get actionTitle(): string {
    const record = this.primary;

    if (!record) {
      return 'Action en attente';
    }

    const expectedImpact = this.asObject(record.expected_impact);
    const expectedRoas = this.toNumber(expectedImpact['expected_roas']);
    const deltaConversions = this.toNumber(expectedImpact['delta_conversions']);

    if (this.isMaintainAction(record)) {
      if ((expectedRoas !== undefined && expectedRoas < 1) || (deltaConversions !== undefined && deltaConversions < 0)) {
        return 'Maintenir sous surveillance';
      }

      return 'Maintenir le budget';
    }

    return record.action_label || record.recommended_action || 'Action en attente';
  }

  get businessAdvice(): string {
    const record = this.primary;

    if (!record) {
      return "Lancer l'audit pour afficher une recommandation.";
    }

    const expectedImpact = this.asObject(record.expected_impact);
    const expectedRoas = this.toNumber(expectedImpact['expected_roas']);
    const deltaConversions = this.toNumber(expectedImpact['delta_conversions']);
    const action = String(record.recommended_action || '').toLowerCase();

    if (action.includes('continue_monitoring') || action.includes('monitoring')) {
      return "Aucun changement budgetaire requis : continuer le monitoring et verifier les KPIs au prochain audit.";
    }

    if (this.isMaintainAction(record)) {
      if (expectedRoas !== undefined && expectedRoas < 1) {
        return "Ne pas scaler maintenant : surveiller 48h et revoir le ciblage, les creatives ou l'offre.";
      }

      if (deltaConversions !== undefined && deltaConversions < 0) {
        return "Maintenir le budget, mais suivre les conversions avant toute hausse.";
      }

      return "Continuer la surveillance avant modification budgetaire.";
    }

    if (action.includes('decrease')) {
      return "Reduire progressivement et verifier que le CPA baisse.";
    }

    if (action.includes('increase')) {
      return "Augmenter par palier avec controle ROAS, CPA et conversions.";
    }

    if (action.includes('reallocate')) {
      return "Reallouer en test controle puis comparer les canaux.";
    }

    if (action.includes('pause')) {
      return "Pause possible seulement apres validation humaine.";
    }

    return "Appliquer progressivement et suivre les KPIs prioritaires.";
  }

  get budgetSignalWidth(): number {
    return Math.min(100, Math.max(8, this.totalSpend / 10));
  }

  get rootCauseLabel(): string {
    return this.primary?.root_cause_label || this.primary?.root_cause || 'Cause a confirmer';
  }

  get campaignGroupLabel(): string {
    const record = this.primary;

    if (!record) {
      return 'Aucune campagne';
    }

    return record.global_campaign_id || record.campaign_id;
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

  displayMoney(value: unknown, fallback = '--'): string {
    const numberValue = Number(value);

    if (!Number.isFinite(numberValue)) {
      return fallback;
    }

    return `${numberValue.toFixed(2)} EUR`;
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

  private anomalyTitle(anomaly: JsonObject): string {
    const raw = String(anomaly['explanation'] || 'Anomalie detectee');
    return raw.split(' | ')[0].replace(/^\[[^\]]+\]\s*/, '');
  }

  private anomalyMeta(anomaly: JsonObject): string {
    const date = anomaly['date'] ? String(anomaly['date']) : 'Date recente';
    const score = this.displayNumber(anomaly['score']);
    const roas = this.displayRoas(anomaly['roas']);

    return `${date} - score ${score} - ROAS ${roas}`;
  }

  private isMaintainAction(record: CampaignOptimizationRecord): boolean {
    const action = String(record.recommended_action || '').toLowerCase();
    const label = String(record.action_label || '').toLowerCase();

    return action.includes('maintain') || label.startsWith('maint');
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

  private averageFromRecords(getter: (record: CampaignOptimizationRecord) => number | undefined): number | undefined {
    const values = this.records
      .map((record) => getter(record))
      .filter((value): value is number => value !== undefined);

    if (!values.length) {
      return undefined;
    }

    return values.reduce((sum, value) => sum + value, 0) / values.length;
  }

  private sumFromRecords(getter: (record: CampaignOptimizationRecord) => number | undefined): number {
    return this.records
      .map((record) => getter(record))
      .filter((value): value is number => value !== undefined)
      .reduce((sum, value) => sum + value, 0);
  }
}
