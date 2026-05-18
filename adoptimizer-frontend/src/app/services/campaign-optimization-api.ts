import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';

import { environment } from '../../environments/environment';

export interface JsonItems<T> {
  items?: T[];
}

export interface CampaignOptimizationRecord {
  id?: number;
  workflow_execution_id?: string;
  global_campaign_id?: string;
  campaign_id: string;
  platform: string;
  health_score?: number;
  health_status?: string;
  anomaly_level?: string;
  anomaly_score?: number;
  n_anomaly_days?: number;
  current_kpis?: unknown;
  predicted_kpis?: unknown;
  trend?: unknown;
  anomaly_types?: unknown;
  top_anomalies?: unknown;
  root_cause?: string;
  root_cause_label?: string;
  causal_confidence?: number;
  recommended_action?: string;
  action_label?: string;
  priority?: string;
  expected_impact?: unknown;
  budget_adjustment?: unknown;
  dashboard_summary?: string;
  source_generated_at?: string;
  created_at?: string;
}

export interface CampaignOptimizationDashboardResponse {
  count: number;
  items: CampaignOptimizationRecord[];
  generated_at: string;
}

@Injectable({
  providedIn: 'root',
})
export class CampaignOptimizationApi {
  private readonly case1WebhookUrl =
    `${environment.apiUrl}/workflows/case1/audit`;
  private readonly dashboardWebhookUrl =
    `${environment.apiUrl}/workflows/case1/dashboard`;

  constructor(private readonly http: HttpClient) {}

  triggerExistingCampaignAudit(): Observable<unknown> {
    return this.http.post(this.case1WebhookUrl, {});
  }

  getOptimizationDashboard(): Observable<CampaignOptimizationDashboardResponse> {
    return this.http.post<CampaignOptimizationDashboardResponse>(this.dashboardWebhookUrl, {});
  }
}
