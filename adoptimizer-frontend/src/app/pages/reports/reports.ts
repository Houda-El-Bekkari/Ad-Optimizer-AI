import { ChangeDetectorRef, Component, OnInit } from '@angular/core';

import {
  AuditReportApi,
  AuditReportCampaign,
  AuditReportData,
} from '../../services/audit-report-api';

@Component({
  selector: 'app-reports',
  templateUrl: './reports.html',
  styleUrls: ['../shared/internal-page.scss', './reports.scss'],
})
export class Reports implements OnInit {
  private readonly auditReportPdfUrl = 'http://127.0.0.1:8000/audit-report/pdf';

  report?: AuditReportData;
  loading = true;
  error = '';

  constructor(
    private readonly api: AuditReportApi,
    private readonly cdr: ChangeDetectorRef,
  ) {}

  ngOnInit(): void {
    this.loadReport();
  }

  get campaigns(): AuditReportCampaign[] {
    return this.report?.campaigns ?? [];
  }

  get executiveSummary() {
    return this.report?.executive_summary;
  }

  get portfolioSummary() {
    return this.report?.portfolio_summary;
  }

  get generatedAtLabel(): string {
    if (!this.report?.generated_at) {
      return 'Audit en attente';
    }

    return new Date(this.report.generated_at).toLocaleString('fr-FR');
  }

  loadReport(): void {
    this.loading = true;
    this.error = '';

    this.api.generateAuditReport().subscribe({
      next: (response) => {
        if (response.status !== 'success' || !response.data) {
          this.error = response.message || "Le rapport d'audit n'a pas pu etre genere.";
          this.report = undefined;
        } else {
          this.report = response.data;
        }

        this.loading = false;
        this.cdr.detectChanges();
      },
      error: () => {
        this.loading = false;
        this.report = undefined;
        this.error =
          "Impossible de joindre FastAPI. Verifiez que uvicorn est lance et que /audit-report est disponible.";
        this.cdr.detectChanges();
      },
    });
  }

  downloadPdf(): void {
    window.open(this.auditReportPdfUrl, '_blank');
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

  displayRoas(value: unknown, fallback = '--'): string {
    const numberValue = Number(value);

    if (!Number.isFinite(numberValue)) {
      return fallback;
    }

    return `${numberValue.toFixed(2)}x`;
  }

  displayPercent(value: unknown, fallback = '--'): string {
    const numberValue = Number(value);

    if (!Number.isFinite(numberValue)) {
      return fallback;
    }

    return `${numberValue.toFixed(1)}%`;
  }

  anomalyTitle(campaign: AuditReportCampaign): string {
    const anomaly = campaign.anomalies?.top_anomalies?.[0];
    const raw = anomaly?.explanation || 'Aucune anomalie prioritaire';

    return raw.split(' | ')[0].replace(/^\[[^\]]+\]\s*/, '');
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
}
