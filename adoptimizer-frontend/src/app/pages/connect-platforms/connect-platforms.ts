import { ChangeDetectorRef, Component } from '@angular/core';
import { Router, RouterLink } from '@angular/router';

import { CampaignOptimizationApi } from '../../services/campaign-optimization-api';

@Component({
  selector: 'app-connect-platforms',
  imports: [RouterLink],
  templateUrl: './connect-platforms.html',
  styleUrl: './connect-platforms.scss',
})
export class ConnectPlatforms {
  isStartingAudit = false;
  auditError = '';

  constructor(
    private readonly campaignOptimizationApi: CampaignOptimizationApi,
    private readonly router: Router,
    private readonly cdr: ChangeDetectorRef,
  ) {}

  startIntelligentAudit(): void {
    if (this.isStartingAudit) {
      return;
    }

    this.isStartingAudit = true;
    this.auditError = '';

    this.campaignOptimizationApi.triggerExistingCampaignAudit().subscribe({
      next: () => {
        this.isStartingAudit = false;
        this.cdr.detectChanges();
        void this.router.navigate(['/ai-audit']);
      },
      error: () => {
        this.isStartingAudit = false;
        this.auditError =
          "Impossible de lancer l'audit. Verifiez que n8n est ouvert et que le workflow est publie.";
        this.cdr.detectChanges();
      },
    });
  }
}
