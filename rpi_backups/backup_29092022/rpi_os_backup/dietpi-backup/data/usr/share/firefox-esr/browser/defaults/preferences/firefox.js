//@line 2 "$SRCDIR/browser/app/profile/firefox.js"
//@line 21 "$SRCDIR/browser/app/profile/firefox.js"
pref("browser.hiddenWindowChromeURL", "chrome://browser/content/hiddenWindowMac.xhtml");
pref("extensions.logging.enabled", false);
pref("extensions.strictCompatibility", false);
pref("extensions.checkCompatibility.temporaryThemeOverride_minAppVersion", "29.0a1");
pref("extensions.webextOptionalPermissionPrompts", true);
pref("extensions.postDownloadThirdPartyPrompt", true);
pref("extensions.getAddons.cache.enabled", true);
pref("extensions.getAddons.get.url", "https://services.addons.mozilla.org/api/v4/addons/search/?guid=%IDS%&lang=%LOCALE%");
pref("extensions.getAddons.search.browseURL", "https://addons.mozilla.org/%LOCALE%/firefox/search?q=%TERMS%&platform=%OS%&appver=%VERSION%");
pref("extensions.getAddons.link.url", "https://addons.mozilla.org/%LOCALE%/firefox/");
pref("extensions.getAddons.langpacks.url", "https://services.addons.mozilla.org/api/v4/addons/language-tools/?app=firefox&type=language&appversion=%VERSION%");
pref("extensions.getAddons.discovery.api_url", "https://services.addons.mozilla.org/api/v4/discovery/?lang=%LOCALE%&edition=%DISTRIBUTION%");
pref("extensions.recommendations.privacyPolicyUrl", "https://www.mozilla.org/privacy/firefox/?utm_source=firefox-browser&utm_medium=firefox-browser&utm_content=privacy-policy-link#addons");
pref("extensions.recommendations.themeRecommendationUrl", "https://color.firefox.com/?utm_source=firefox-browser&utm_medium=firefox-browser&utm_content=theme-footer-link");
pref("extensions.update.autoUpdateDefault", true);
pref("extensions.systemAddon.update.url", "https://aus5.mozilla.org/update/3/SystemAddons/%VERSION%/%BUILD_ID%/%BUILD_TARGET%/%LOCALE%/%CHANNEL%/%OS_VERSION%/%DISTRIBUTION%/%DISTRIBUTION_VERSION%/update.xml");
pref("extensions.systemAddon.update.enabled", true);
pref("extensions.autoDisableScopes", 3);
pref("extensions.startupScanScopes", 0);
pref("extensions.geckoProfiler.acceptedExtensionIds", "geckoprofiler@mozilla.com,quantum-foxfooding@mozilla.com,raptor@mozilla.org");
pref("extensions.webextensions.remote", true);
pref("extensions.langpacks.signatures.required", true);
pref("xpinstall.signatures.required", true);
pref("xpinstall.signatures.devInfoURL", "https://wiki.mozilla.org/Addons/Extension_Signing");
pref("devtools.storage.extensionStorage.enabled", true);
pref("browser.dictionaries.download.url", "https://addons.mozilla.org/%LOCALE%/firefox/language-tools/");
pref("app.update.checkInstallTime", true);
pref("app.update.timerMinimumDelay", 120);
pref("app.update.timerFirstInterval", 30000);
pref("app.update.log", false);
pref("app.update.log.file", false);
pref("app.update.backgroundMaxErrors", 10);
pref("app.update.link.updateAvailableWhatsNew", "update-available-whats-new");
pref("app.update.link.updateManualWhatsNew", "update-manual-whats-new");
pref("app.update.download.promptMaxAttempts", 2);
pref("app.update.elevation.promptMaxAttempts", 2);
//@line 137 "$SRCDIR/browser/app/profile/firefox.js"
pref("app.update.notifyDuringDownload", false);
//@line 149 "$SRCDIR/browser/app/profile/firefox.js"
  pref("app.update.auto", true);
//@line 151 "$SRCDIR/browser/app/profile/firefox.js"
pref("app.update.staging.enabled", true);
//@line 169 "$SRCDIR/browser/app/profile/firefox.js"
//@line 175 "$SRCDIR/browser/app/profile/firefox.js"
pref("app.update.langpack.enabled", true);
//@line 184 "$SRCDIR/browser/app/profile/firefox.js"
//@line 193 "$SRCDIR/browser/app/profile/firefox.js"
//@line 195 "$SRCDIR/browser/app/profile/firefox.js"
  pref("toolkit.backgroundtasks.defaultTimeoutSec", 600);
//@line 199 "$SRCDIR/browser/app/profile/firefox.js"
pref("extensions.update.enabled", true);
pref("extensions.update.url", "https://versioncheck.addons.mozilla.org/update/VersionCheck.php?reqVersion=%REQ_VERSION%&id=%ITEM_ID%&version=%ITEM_VERSION%&maxAppVersion=%ITEM_MAXAPPVERSION%&status=%ITEM_STATUS%&appID=%APP_ID%&appVersion=%APP_VERSION%&appOS=%APP_OS%&appABI=%APP_ABI%&locale=%APP_LOCALE%&currentAppVersion=%CURRENT_APP_VERSION%&updateType=%UPDATE_TYPE%&compatMode=%COMPATIBILITY_MODE%");
pref("extensions.update.background.url", "https://versioncheck-bg.addons.mozilla.org/update/VersionCheck.php?reqVersion=%REQ_VERSION%&id=%ITEM_ID%&version=%ITEM_VERSION%&maxAppVersion=%ITEM_MAXAPPVERSION%&status=%ITEM_STATUS%&appID=%APP_ID%&appVersion=%APP_VERSION%&appOS=%APP_OS%&appABI=%APP_ABI%&locale=%APP_LOCALE%&currentAppVersion=%CURRENT_APP_VERSION%&updateType=%UPDATE_TYPE%&compatMode=%COMPATIBILITY_MODE%");
pref("extensions.update.interval", 86400);  // Check for updates to Extensions and
pref("lightweightThemes.getMoreURL", "https://addons.mozilla.org/%LOCALE%/firefox/themes");
//@line 217 "$SRCDIR/browser/app/profile/firefox.js"
  pref("browser.eme.ui.enabled", false);
//@line 219 "$SRCDIR/browser/app/profile/firefox.js"
pref("browser.uitour.enabled", true);
pref("browser.uitour.loglevel", "Error");
pref("browser.uitour.requireSecure", true);
pref("browser.uitour.themeOrigin", "https://addons.mozilla.org/%LOCALE%/firefox/themes/");
pref("browser.uitour.url", "https://www.mozilla.org/%LOCALE%/firefox/%VERSION%/tour/");
pref("browser.uitour.surveyDuration", 7200);
pref("keyword.enabled", true);
pref("browser.fixup.domainwhitelist.localhost", true);
pref("browser.fixup.domainsuffixwhitelist.test", true);
pref("browser.fixup.domainsuffixwhitelist.example", true);
pref("browser.fixup.domainsuffixwhitelist.invalid", true);
pref("browser.fixup.domainsuffixwhitelist.localhost", true);
pref("browser.fixup.domainsuffixwhitelist.internal", true);
pref("browser.fixup.domainsuffixwhitelist.local", true);
pref("browser.fixup.dns_first_for_single_words", false);
//@line 250 "$SRCDIR/browser/app/profile/firefox.js"
  pref("general.autoScroll", false);
//@line 254 "$SRCDIR/browser/app/profile/firefox.js"
pref("browser.uidensity", 0);
pref("browser.touchmode.auto", true);
pref("browser.compactmode.show", false);
pref("browser.shell.checkDefaultBrowser", true);
pref("browser.shell.shortcutFavicons",true);
pref("browser.shell.mostRecentDateSetAsDefault", "");
pref("browser.shell.skipDefaultBrowserCheckOnFirstRun", true);
pref("browser.shell.didSkipDefaultBrowserCheckOnFirstRun", false);
pref("browser.shell.defaultBrowserCheckCount", 0);
//@line 284 "$SRCDIR/browser/app/profile/firefox.js"
pref("browser.startup.page",                1);
pref("browser.startup.homepage",            "about:home");
//@line 293 "$SRCDIR/browser/app/profile/firefox.js"
pref("browser.startup.homepage.abouthome_cache.enabled", false);
//@line 295 "$SRCDIR/browser/app/profile/firefox.js"
pref("browser.startup.homepage.abouthome_cache.loglevel", "Warn");
pref("browser.startup.firstrunSkipsHomepage", true);
pref("browser.startup.couldRestoreSession.count", 0);
//@line 309 "$SRCDIR/browser/app/profile/firefox.js"
  pref("browser.startup.blankWindow", false);
//@line 311 "$SRCDIR/browser/app/profile/firefox.js"
//@line 317 "$SRCDIR/browser/app/profile/firefox.js"
pref("browser.startup.upgradeDialog.enabled", false);
pref("toolkit.lazyHiddenWindow", true);
pref("browser.chrome.site_icons", true);
pref("browser.warnOnQuit", true);
//@line 333 "$SRCDIR/browser/app/profile/firefox.js"
  pref("browser.warnOnQuitShortcut", true);
//@line 335 "$SRCDIR/browser/app/profile/firefox.js"
//@line 340 "$SRCDIR/browser/app/profile/firefox.js"
  pref("browser.fullscreen.autohide", true);
//@line 342 "$SRCDIR/browser/app/profile/firefox.js"
pref("browser.overlink-delay", 80);
pref("browser.theme.colorway-closet", false);
pref("browser.urlbar.ctrlCanonizesURLs", true);
pref("browser.urlbar.accessibility.tabToSearch.announceResults", true);
pref("browser.urlbar.autoFill", true);
pref("browser.urlbar.autoFill.adaptiveHistory.enabled", false);
pref("browser.urlbar.autoFill.adaptiveHistory.minCharsThreshold", 0);
pref("browser.urlbar.speculativeConnect.enabled", true);
pref("browser.urlbar.filter.javascript", true);
pref("browser.urlbar.loglevel", "Error");
pref("browser.urlbar.maxRichResults", 10);
pref("browser.urlbar.maxHistoricalSearchSuggestions", 2);
pref("browser.urlbar.suggest.bookmark",             true);
pref("browser.urlbar.suggest.history",              true);
pref("browser.urlbar.suggest.openpage",             true);
pref("browser.urlbar.suggest.remotetab",            true);
pref("browser.urlbar.suggest.searches",             true);
pref("browser.urlbar.suggest.topsites",             true);
pref("browser.urlbar.suggest.engines",              true);
pref("browser.urlbar.suggest.calculator",           false);
pref("browser.urlbar.suggest.bestmatch", true, sticky);
pref("browser.urlbar.suggest.quicksuggest.nonsponsored", false, sticky);
pref("browser.urlbar.suggest.quicksuggest.sponsored", false, sticky);
pref("browser.urlbar.quicksuggest.dataCollection.enabled", false, sticky);
pref("browser.urlbar.quicksuggest.enabled", false);
pref("browser.urlbar.quicksuggest.shouldShowOnboardingDialog", true);
pref("browser.urlbar.quicksuggest.showOnboardingDialogAfterNRestarts", 0);
pref("browser.urlbar.quicksuggest.sponsoredIndex", -1);
pref("browser.urlbar.quicksuggest.nonSponsoredIndex", -1);
pref("browser.urlbar.quicksuggest.remoteSettings.enabled", true);
pref("browser.urlbar.quicksuggest.allowPositionInSuggestions", true);
pref("browser.urlbar.quicksuggest.impressionCaps.nonSponsoredEnabled", false);
pref("browser.urlbar.quicksuggest.impressionCaps.sponsoredEnabled", false);
pref("browser.urlbar.quicksuggest.blockingEnabled", false);
//@line 457 "$SRCDIR/browser/app/profile/firefox.js"
pref("browser.urlbar.unitConversion.enabled", false);
//@line 459 "$SRCDIR/browser/app/profile/firefox.js"
pref("browser.urlbar.showSearchSuggestionsFirst", true);
pref("browser.urlbar.maxCharsForSearchSuggestions", 100);
pref("browser.urlbar.trimURLs", true);
pref("browser.urlbar.decodeURLsOnCopy", false);
pref("browser.urlbar.switchTabs.adoptIntoActiveWindow", false);
pref("browser.urlbar.openintab", false);
pref("browser.urlbar.richSuggestions.tail", true);
pref("browser.urlbar.sponsoredTopSites", false);
pref("browser.urlbar.update2.emptySearchBehavior", 0);
pref("browser.urlbar.shortcuts.bookmarks", true);
pref("browser.urlbar.shortcuts.tabs", true);
pref("browser.urlbar.shortcuts.history", true);
pref("browser.urlbar.eventTelemetry.enabled", false);
pref("browser.urlbar.extension.timeout", 400);
pref("browser.urlbar.dnsResolveSingleWordsAfterSearch", 1);
pref("browser.urlbar.keepPanelOpenDuringImeComposition", false);
pref("browser.urlbar.groupLabels.enabled", true);
pref("browser.urlbar.merino.enabled", true);
pref("browser.urlbar.merino.endpointURL", "https://merino.services.mozilla.com/api/v1/suggest");
pref("browser.urlbar.merino.timeoutMs", 200);
pref("browser.urlbar.merino.providers", "");
pref("browser.urlbar.merino.clientVariants", "");
pref("browser.urlbar.bestMatch.enabled", false);
pref("browser.urlbar.bestMatch.blockingEnabled", false);
pref("browser.altClickSave", false);
pref("browser.download.loglevel", "Error");
pref("browser.download.saveLinkAsFilenameTimeout", 4000);
pref("browser.download.useDownloadDir", true);
pref("browser.download.folderList", 1);
pref("browser.download.manager.addToRecentDocs", true);
pref("browser.download.manager.resumeOnWakeDelay", 10000);
pref("browser.download.animateNotifications", true);
pref("browser.download.panel.shown", false);
pref("browser.download.openInSystemViewerContextMenuItem", true);
pref("browser.download.alwaysOpenInSystemViewerContextMenuItem", true);
pref("browser.download.viewableInternally.enabledTypes", "xml,svg,webp,avif,jxl");
pref("browser.download.autohideButton", true);
pref("browser.download.alwaysOpenPanel", true);
pref("browser.download.clearHistoryOnDelete", 0);
//@line 595 "$SRCDIR/browser/app/profile/firefox.js"
  pref("browser.helperApps.deleteTempFileOnExit", true);
//@line 597 "$SRCDIR/browser/app/profile/firefox.js"
pref("browser.helperApps.showOpenOptionForPdfJS", true);
pref("browser.helperApps.showOpenOptionForViewableInternally", true);
pref("browser.search.searchEngineRemoval", "https://support.mozilla.org/1/firefox/%VERSION%/%OS%/%LOCALE%/search-engine-removal");
pref("browser.search.searchEnginesURL",      "https://addons.mozilla.org/%LOCALE%/firefox/search-engines/");
pref("browser.search.openintab", false);
pref("browser.search.context.loadInBackground", false);
pref("browser.search.hiddenOneOffs", "");
pref("browser.search.widget.inNavBar", false);
pref("browser.search.separatePrivateDefault.ui.enabled", false);
pref("browser.search.separatePrivateDefault.ui.banner.max", 0);
pref("browser.privatebrowsing.vpnpromourl", "https://vpn.mozilla.org/?utm_source=firefox-browser&utm_medium=firefox-%CHANNEL%-browser&utm_campaign=private-browsing-vpn-link");
pref("browser.privacySegmentation.enabled", false);
pref("browser.theme.dark-private-windows", false);
pref("browser.privacySegmentation.preferences.show", false);
pref("browser.privacySegmentation.preferences.learnMoreURLSuffix", "data-features");
pref("browser.sessionhistory.max_entries", 50);
pref("permissions.manager.defaultsUrl", "resource://app/defaults/permissions");
pref("permissions.default.camera", 0);
pref("permissions.default.microphone", 0);
pref("permissions.default.geo", 0);
pref("permissions.default.xr", 0);
pref("permissions.default.desktop-notification", 0);
pref("permissions.default.shortcuts", 0);
pref("permissions.desktop-notification.postPrompt.enabled", true);
pref("permissions.desktop-notification.notNow.enabled", false);
pref("permissions.fullscreen.allowed", false);
pref("browser.link.open_newwindow", 3);
pref("browser.link.open_newwindow.override.external", -1);
pref("browser.link.open_newwindow.restriction", 2);
//@line 687 "$SRCDIR/browser/app/profile/firefox.js"
  pref("browser.link.open_newwindow.disabled_in_fullscreen", false);
//@line 689 "$SRCDIR/browser/app/profile/firefox.js"
pref("browser.tabs.closeTabByDblclick", false);
pref("browser.tabs.closeWindowWithLastTab", true);
pref("browser.tabs.allowTabDetach", true);
pref("browser.tabs.insertRelatedAfterCurrent", true);
pref("browser.tabs.insertAfterCurrent", false);
pref("browser.tabs.warnOnClose", false);
pref("browser.tabs.warnOnCloseOtherTabs", true);
pref("browser.tabs.warnOnOpen", true);
pref("browser.tabs.maxOpenBeforeWarn", 15);
pref("browser.tabs.loadInBackground", true);
pref("browser.tabs.opentabfor.middleclick", true);
pref("browser.tabs.loadDivertedInBackground", false);
pref("browser.tabs.loadBookmarksInBackground", false);
pref("browser.tabs.loadBookmarksInTabs", false);
pref("browser.tabs.tabClipWidth", 140);
pref("browser.tabs.tabMinWidth", 76);
pref("browser.tabs.secondaryTextUnsupportedLocales", "ar,bn,bo,ckb,fa,gu,he,hi,ja,km,kn,ko,lo,mr,my,ne,pa,si,ta,te,th,ur,zh");
pref("browser.tabs.tabmanager.enabled", false);
pref("browser.tabs.selectOwnerOnClose", true);
pref("browser.tabs.delayHidingAudioPlayingIconMS", 3000);
pref("browser.tabs.remote.separatePrivilegedContentProcess", true);
//@line 743 "$SRCDIR/browser/app/profile/firefox.js"
pref("browser.tabs.remote.separatePrivilegedMozillaWebContentProcess", true);
//@line 752 "$SRCDIR/browser/app/profile/firefox.js"
pref("browser.tabs.tooltipsShowPidAndActiveness", false);
//@line 754 "$SRCDIR/browser/app/profile/firefox.js"
pref("browser.tabs.firefox-view", false);
pref("security.allow_eval_with_system_principal", false);
pref("security.allow_eval_in_parent_process", false);
pref("security.allow_parent_unrestricted_js_loads", false);
pref("browser.tabs.unloadOnLowMemory", true);
pref("browser.tabs.min_inactive_duration_before_unload", 600000);
//@line 804 "$SRCDIR/browser/app/profile/firefox.js"
pref("browser.ctrlTab.sortByRecentlyUsed", false);
pref("browser.bookmarks.autoExportHTML",          false);
pref("browser.bookmarks.max_backups",             15);
pref("browser.bookmarks.openInTabClosesMenu", true);
pref("browser.bookmarks.defaultLocation", "toolbar");
pref("dom.disable_open_during_load",              true);
pref("dom.disable_window_move_resize",            false);
pref("dom.disable_window_flip",                   true);
pref("privacy.popups.policy",               1);
pref("privacy.popups.usecustom",            true);
pref("privacy.popups.showBrowserMessage",   true);
pref("privacy.item.cookies",                false);
pref("privacy.clearOnShutdown.history",     true);
pref("privacy.clearOnShutdown.formdata",    true);
pref("privacy.clearOnShutdown.downloads",   true);
pref("privacy.clearOnShutdown.cookies",     true);
pref("privacy.clearOnShutdown.cache",       true);
pref("privacy.clearOnShutdown.sessions",    true);
pref("privacy.clearOnShutdown.offlineApps", false);
pref("privacy.clearOnShutdown.siteSettings", false);
pref("privacy.clearOnShutdown.openWindows", false);
pref("privacy.cpd.history",                 true);
pref("privacy.cpd.formdata",                true);
pref("privacy.cpd.passwords",               false);
pref("privacy.cpd.downloads",               true);
pref("privacy.cpd.cookies",                 true);
pref("privacy.cpd.cache",                   true);
pref("privacy.cpd.sessions",                true);
pref("privacy.cpd.offlineApps",             false);
pref("privacy.cpd.siteSettings",            false);
pref("privacy.cpd.openWindows",             false);
pref("privacy.history.custom",              false);
pref("privacy.sanitize.timeSpan", 1);
pref("privacy.sanitize.migrateFx3Prefs",    false);
pref("privacy.panicButton.enabled",         true);
pref("privacy.temporary_permission_expire_time_ms",  3600000);
pref("network.proxy.share_proxy_settings",  false); // use the same proxy settings for all protocols
pref("browser.gesture.swipe.left", "Browser:BackOrBackDuplicate");
pref("browser.gesture.swipe.right", "Browser:ForwardOrForwardDuplicate");
pref("browser.gesture.swipe.up", "cmd_scrollTop");
pref("browser.gesture.swipe.down", "cmd_scrollBottom");
pref("browser.gesture.pinch.latched", false);
pref("browser.gesture.pinch.threshold", 25);
//@line 896 "$SRCDIR/browser/app/profile/firefox.js"
  pref("browser.gesture.pinch.out", "cmd_fullZoomEnlarge");
  pref("browser.gesture.pinch.in", "cmd_fullZoomReduce");
  pref("browser.gesture.pinch.out.shift", "cmd_fullZoomReset");
  pref("browser.gesture.pinch.in.shift", "cmd_fullZoomReset");
//@line 908 "$SRCDIR/browser/app/profile/firefox.js"
pref("browser.gesture.twist.latched", false);
pref("browser.gesture.twist.threshold", 0);
pref("browser.gesture.twist.right", "cmd_gestureRotateRight");
pref("browser.gesture.twist.left", "cmd_gestureRotateLeft");
pref("browser.gesture.twist.end", "cmd_gestureRotateEnd");
//@line 914 "$SRCDIR/browser/app/profile/firefox.js"
  pref("browser.gesture.tap", "cmd_fullZoomReset");
//@line 918 "$SRCDIR/browser/app/profile/firefox.js"
pref("browser.history_swipe_animation.disabled", false);
//@line 944 "$SRCDIR/browser/app/profile/firefox.js"
  pref("mousewheel.with_shift.action", 4);
  pref("mousewheel.with_alt.action", 2);
  pref("mousewheel.with_meta.action", 1); // win key on Win, Super/Hyper on Linux
//@line 954 "$SRCDIR/browser/app/profile/firefox.js"
pref("mousewheel.with_control.action",3);
pref("mousewheel.with_win.action", 1);
pref("browser.xul.error_pages.expert_bad_cert", false);
pref("browser.xul.error_pages.show_safe_browsing_details_on_load", false);
pref("network.captive-portal-service.enabled", true);
pref("network.manage-offline-status", true);
pref("network.protocol-handler.external.mailto", true); // for mail
pref("network.protocol-handler.external.news", true);   // for news
pref("network.protocol-handler.external.snews", true);  // for secure news
pref("network.protocol-handler.external.nntp", true);   // also news
//@line 974 "$SRCDIR/browser/app/profile/firefox.js"
pref("network.protocol-handler.warn-external.mailto", false);
pref("network.protocol-handler.warn-external.news", false);
pref("network.protocol-handler.warn-external.snews", false);
pref("network.protocol-handler.warn-external.nntp", false);
//@line 983 "$SRCDIR/browser/app/profile/firefox.js"
pref("network.protocol-handler.expose-all", true);
pref("network.protocol-handler.expose.mailto", false);
pref("network.protocol-handler.expose.news", false);
pref("network.protocol-handler.expose.snews", false);
pref("network.protocol-handler.expose.nntp", false);
pref("accessibility.typeaheadfind", false);
pref("accessibility.typeaheadfind.timeout", 5000);
pref("accessibility.typeaheadfind.linksonly", false);
pref("accessibility.typeaheadfind.flashBar", 1);
pref("accessibility.support.url", "https://support.mozilla.org/%LOCALE%/kb/accessibility-services");
pref("accessibility.indicator.enabled", false);
pref("plugins.testmode", false);
pref("plugins.show_infobar", false);
//@line 1011 "$SRCDIR/browser/app/profile/firefox.js"
  pref("plugin.default.state", 1);
//@line 1013 "$SRCDIR/browser/app/profile/firefox.js"
pref("plugins.flashBlock.enabled", true);
pref("plugins.favorfallback.mode", "follow-ctp");
pref("plugins.favorfallback.rules", "nosrc,video");
//@line 1028 "$SRCDIR/browser/app/profile/firefox.js"
  pref("browser.preferences.instantApply", true);
//@line 1030 "$SRCDIR/browser/app/profile/firefox.js"
pref("browser.preferences.search", true);
//@line 1036 "$SRCDIR/browser/app/profile/firefox.js"
pref("browser.preferences.experimental", false);
//@line 1038 "$SRCDIR/browser/app/profile/firefox.js"
pref("browser.preferences.moreFromMozilla", true);
pref("browser.preferences.experimental.hidden", false);
pref("browser.preferences.defaultPerformanceSettings.enabled", true);
pref("browser.proton.toolbar.version", 0);
pref("browser.backspace_action", 2);
pref("intl.regional_prefs.use_os_locales", false);
pref("layout.spellcheckDefault", 1);
pref("browser.send_pings", false);
pref("browser.geolocation.warning.infoURL", "https://www.mozilla.org/%LOCALE%/firefox/geolocation/");
pref("browser.xr.warning.infoURL", "https://www.mozilla.org/%LOCALE%/firefox/xr/");
pref("browser.sessionstore.resume_from_crash", true);
pref("browser.sessionstore.resume_session_once", false);
pref("browser.sessionstore.resuming_after_os_restart", false);
pref("browser.sessionstore.interval.idle", 3600000); // 1h
pref("browser.sessionstore.idleDelay", 180000); // 3 minutes
pref("browser.sessionstore.privacy_level", 0);
pref("browser.sessionstore.max_tabs_undo", 25);
pref("browser.sessionstore.max_windows_undo", 3);
pref("browser.sessionstore.max_resumed_crashes", 1);
pref("browser.sessionstore.max_serialize_back", 10);
pref("browser.sessionstore.max_serialize_forward", -1);
pref("browser.sessionstore.restore_on_demand", true);
pref("browser.sessionstore.restore_hidden_tabs", false);
pref("browser.sessionstore.restore_pinned_tabs_on_demand", false);
pref("browser.sessionstore.upgradeBackup.latestBuildID", "");
pref("browser.sessionstore.upgradeBackup.maxUpgradeBackups", 3);
pref("browser.sessionstore.debug", false);
pref("browser.sessionstore.cleanup.forget_closed_after", 1209600000);
pref("browser.quitShortcut.disabled", false);
pref("accessibility.blockautorefresh", false);
pref("places.history.enabled", true);
pref("places.search.matchDiacritics", false);
pref("places.frecency.numVisits", 10);
pref("places.frecency.firstBucketCutoff", 4);
pref("places.frecency.secondBucketCutoff", 14);
pref("places.frecency.thirdBucketCutoff", 31);
pref("places.frecency.fourthBucketCutoff", 90);
pref("places.frecency.firstBucketWeight", 100);
pref("places.frecency.secondBucketWeight", 70);
pref("places.frecency.thirdBucketWeight", 50);
pref("places.frecency.fourthBucketWeight", 30);
pref("places.frecency.defaultBucketWeight", 10);
pref("places.frecency.embedVisitBonus", 0);
pref("places.frecency.framedLinkVisitBonus", 0);
pref("places.frecency.linkVisitBonus", 100);
pref("places.frecency.typedVisitBonus", 2000);
pref("places.frecency.bookmarkVisitBonus", 75);
pref("places.frecency.redirectSourceVisitBonus", 25);
pref("places.frecency.downloadVisitBonus", 0);
pref("places.frecency.permRedirectVisitBonus", 50);
pref("places.frecency.tempRedirectVisitBonus", 40);
pref("places.frecency.reloadVisitBonus", 0);
pref("places.frecency.defaultVisitBonus", 0);
pref("places.frecency.unvisitedBookmarkBonus", 140);
pref("places.frecency.unvisitedTypedBonus", 200);
pref("places.forgetThisSite.clearByBaseDomain", true);
pref("browser.places.speculativeConnect.enabled", true);
pref("browser.ssl_override_behavior", 2);
pref("browser.zoom.full", true);
pref("browser.zoom.siteSpecific", true);
pref("browser.zoom.updateBackgroundTabs", true);
pref("breakpad.reportURL", "https://crash-stats.mozilla.org/report/index/");
pref("toolkit.datacollection.infoURL",
     "https://www.mozilla.org/legal/privacy/firefox.html");
pref("toolkit.crashreporter.infoURL",
     "https://www.mozilla.org/legal/privacy/firefox.html#crash-reporter");
pref("app.support.baseURL", "https://support.mozilla.org/1/firefox/%VERSION%/%OS%/%LOCALE%/");
pref("app.feedback.baseURL", "https://ideas.mozilla.org/");
pref("security.alternate_certificate_error_page", "certerror");
pref("security.certerrors.recordEventTelemetry", true);
pref("security.certerrors.permanentOverride", true);
pref("security.certerrors.mitm.priming.enabled", true);
pref("security.certerrors.mitm.priming.endpoint", "https://mitmdetection.services.mozilla.com/");
pref("security.certerrors.mitm.auto_enable_enterprise_roots", true);
pref("browser.bookmarks.editDialog.showForNewBookmarks", true);
pref("browser.bookmarks.editDialog.firstEditField", "namePicker");
pref("browser.bookmarks.editDialog.maxRecentFolders", 7);
pref("browser.bookmarks.editDialog.delayedApply.enabled", false);
pref("dom.ipc.shims.enabledWarnings", false);
//@line 1243 "$SRCDIR/browser/app/profile/firefox.js"
//@line 1269 "$SRCDIR/browser/app/profile/firefox.js"
//@line 1271 "$SRCDIR/browser/app/profile/firefox.js"
  pref("security.sandbox.content.level", 4);
  pref("security.sandbox.socket.process.level", 1);
  pref("security.sandbox.content.write_path_whitelist", "");
  pref("security.sandbox.content.read_path_whitelist", "");
  pref("security.sandbox.content.syscall_whitelist", "");
//@line 1294 "$SRCDIR/browser/app/profile/firefox.js"
//@line 1298 "$SRCDIR/browser/app/profile/firefox.js"
//@line 1300 "$SRCDIR/browser/app/profile/firefox.js"
  pref("security.sandbox.content.tempDirSuffix", "");
  pref("security.sandbox.plugin.tempDirSuffix", "");
//@line 1312 "$SRCDIR/browser/app/profile/firefox.js"
//@line 1323 "$SRCDIR/browser/app/profile/firefox.js"
//@line 1335 "$SRCDIR/browser/app/profile/firefox.js"
pref("services.sync.prefs.sync.accessibility.blockautorefresh", true);
pref("services.sync.prefs.sync.accessibility.browsewithcaret", true);
pref("services.sync.prefs.sync.accessibility.typeaheadfind", true);
pref("services.sync.prefs.sync.accessibility.typeaheadfind.linksonly", true);
pref("services.sync.prefs.sync.addons.ignoreUserEnabledChanges", true);
pref("services.sync.prefs.sync.app.shield.optoutstudies.enabled", true);
pref("services.sync.prefs.sync.browser.contentblocking.category", true);
pref("services.sync.prefs.sync.browser.contentblocking.features.strict", true);
pref("services.sync.prefs.sync.browser.crashReports.unsubmittedCheck.autoSubmit2", true);
pref("services.sync.prefs.sync.browser.ctrlTab.sortByRecentlyUsed", true);
pref("services.sync.prefs.sync.browser.discovery.enabled", true);
pref("services.sync.prefs.sync.browser.download.useDownloadDir", true);
pref("services.sync.prefs.sync.browser.formfill.enable", true);
pref("services.sync.prefs.sync.browser.link.open_newwindow", true);
pref("services.sync.prefs.sync.browser.menu.showViewImageInfo", true);
pref("services.sync.prefs.sync.browser.newtabpage.activity-stream.asrouter.userprefs.cfr.addons", true);
pref("services.sync.prefs.sync.browser.newtabpage.activity-stream.asrouter.userprefs.cfr.features", true);
pref("services.sync.prefs.sync.browser.newtabpage.activity-stream.showSearch", true);
pref("services.sync.prefs.sync.browser.newtabpage.activity-stream.showSponsored", true);
pref("services.sync.prefs.sync.browser.newtabpage.activity-stream.showSponsoredTopSites", true);
pref("services.sync.prefs.sync.browser.newtabpage.activity-stream.feeds.topsites", true);
pref("services.sync.prefs.sync.browser.newtabpage.activity-stream.topSitesRows", true);
pref("services.sync.prefs.sync.browser.newtabpage.activity-stream.feeds.snippets", true);
pref("services.sync.prefs.sync.browser.newtabpage.activity-stream.feeds.section.topstories", true);
pref("services.sync.prefs.sync.browser.newtabpage.activity-stream.section.topstories.rows", true);
pref("services.sync.prefs.sync.browser.newtabpage.activity-stream.feeds.section.highlights", true);
pref("services.sync.prefs.sync.browser.newtabpage.activity-stream.section.highlights.includeVisited", true);
pref("services.sync.prefs.sync.browser.newtabpage.activity-stream.section.highlights.includeBookmarks", true);
pref("services.sync.prefs.sync.browser.newtabpage.activity-stream.section.highlights.includeDownloads", true);
pref("services.sync.prefs.sync.browser.newtabpage.activity-stream.section.highlights.includePocket", true);
pref("services.sync.prefs.sync.browser.newtabpage.activity-stream.section.highlights.rows", true);
pref("services.sync.prefs.sync.browser.newtabpage.enabled", true);
pref("services.sync.prefs.sync.browser.newtabpage.pinned", true);
pref("services.sync.prefs.sync.browser.offline-apps.notify", true);
pref("services.sync.prefs.sync.browser.safebrowsing.downloads.enabled", true);
pref("services.sync.prefs.sync.browser.safebrowsing.downloads.remote.block_potentially_unwanted", true);
pref("services.sync.prefs.sync.browser.safebrowsing.malware.enabled", true);
pref("services.sync.prefs.sync.browser.safebrowsing.phishing.enabled", true);
pref("services.sync.prefs.sync.browser.search.update", true);
pref("services.sync.prefs.sync.browser.search.widget.inNavBar", true);
pref("services.sync.prefs.sync.browser.startup.homepage", true);
pref("services.sync.prefs.sync.browser.startup.page", true);
pref("services.sync.prefs.sync.browser.tabs.loadInBackground", true);
pref("services.sync.prefs.sync.browser.tabs.warnOnClose", true);
pref("services.sync.prefs.sync.browser.tabs.warnOnOpen", true);
pref("services.sync.prefs.sync.browser.taskbar.previews.enable", true);
pref("services.sync.prefs.sync.browser.urlbar.maxRichResults", true);
pref("services.sync.prefs.sync.browser.urlbar.showSearchSuggestionsFirst", true);
pref("services.sync.prefs.sync.browser.urlbar.suggest.bookmark", true);
pref("services.sync.prefs.sync.browser.urlbar.suggest.history", true);
pref("services.sync.prefs.sync.browser.urlbar.suggest.openpage", true);
pref("services.sync.prefs.sync.browser.urlbar.suggest.searches", true);
pref("services.sync.prefs.sync.browser.urlbar.suggest.topsites", true);
pref("services.sync.prefs.sync.browser.urlbar.suggest.engines", true);
pref("services.sync.prefs.sync.dom.disable_open_during_load", true);
pref("services.sync.prefs.sync.dom.disable_window_flip", true);
pref("services.sync.prefs.sync.dom.disable_window_move_resize", true);
pref("services.sync.prefs.sync.dom.event.contextmenu.enabled", true);
pref("services.sync.prefs.sync.dom.security.https_only_mode", true);
pref("services.sync.prefs.sync.dom.security.https_only_mode_ever_enabled", true);
pref("services.sync.prefs.sync.dom.security.https_only_mode_ever_enabled_pbm", true);
pref("services.sync.prefs.sync.dom.security.https_only_mode_pbm", true);
pref("services.sync.prefs.sync.extensions.update.enabled", true);
pref("services.sync.prefs.sync.extensions.activeThemeID", true);
pref("services.sync.prefs.sync.general.autoScroll", true);
pref("services.sync.prefs.sync.general.smoothScroll", true);
pref("services.sync.prefs.sync.intl.accept_languages", true);
pref("services.sync.prefs.sync.intl.regional_prefs.use_os_locales", true);
pref("services.sync.prefs.sync.layout.spellcheckDefault", true);
pref("services.sync.prefs.sync.media.autoplay.default", true);
pref("services.sync.prefs.sync.media.eme.enabled", true);
pref("services.sync.prefs.sync.media.videocontrols.picture-in-picture.video-toggle.enabled", true);
pref("services.sync.prefs.sync.network.cookie.cookieBehavior", true);
pref("services.sync.prefs.sync.network.cookie.lifetimePolicy", true);
pref("services.sync.prefs.sync.network.cookie.thirdparty.sessionOnly", true);
pref("services.sync.prefs.sync.permissions.default.image", true);
pref("services.sync.prefs.sync.pref.downloads.disable_button.edit_actions", true);
pref("services.sync.prefs.sync.pref.privacy.disable_button.cookie_exceptions", true);
pref("services.sync.prefs.sync.privacy.clearOnShutdown.cache", true);
pref("services.sync.prefs.sync.privacy.clearOnShutdown.cookies", true);
pref("services.sync.prefs.sync.privacy.clearOnShutdown.downloads", true);
pref("services.sync.prefs.sync.privacy.clearOnShutdown.formdata", true);
pref("services.sync.prefs.sync.privacy.clearOnShutdown.history", true);
pref("services.sync.prefs.sync.privacy.clearOnShutdown.offlineApps", true);
pref("services.sync.prefs.sync.privacy.clearOnShutdown.sessions", true);
pref("services.sync.prefs.sync.privacy.clearOnShutdown.siteSettings", true);
pref("services.sync.prefs.sync.privacy.donottrackheader.enabled", true);
pref("services.sync.prefs.sync.privacy.sanitize.sanitizeOnShutdown", true);
pref("services.sync.prefs.sync.privacy.trackingprotection.enabled", true);
pref("services.sync.prefs.sync.privacy.trackingprotection.cryptomining.enabled", true);
pref("services.sync.prefs.sync.privacy.trackingprotection.fingerprinting.enabled", true);
pref("services.sync.prefs.sync.privacy.trackingprotection.pbmode.enabled", true);
pref("services.sync.prefs.sync.privacy.reduceTimerPrecision", true);
pref("services.sync.prefs.sync.privacy.resistFingerprinting.reduceTimerPrecision.microseconds", true);
pref("services.sync.prefs.sync.privacy.resistFingerprinting.reduceTimerPrecision.jitter", true);
pref("services.sync.prefs.sync.privacy.userContext.enabled", true);
pref("services.sync.prefs.sync.privacy.userContext.newTabContainerOnLeftClick.enabled", true);
pref("services.sync.prefs.sync.security.default_personal_cert", true);
pref("services.sync.prefs.sync.services.sync.syncedTabs.showRemoteIcons", true);
pref("services.sync.prefs.sync.signon.autofillForms", true);
pref("services.sync.prefs.sync.signon.generation.enabled", true);
pref("services.sync.prefs.sync.signon.management.page.breach-alerts.enabled", true);
pref("services.sync.prefs.sync.signon.rememberSignons", true);
pref("services.sync.prefs.sync.spellchecker.dictionary", true);
pref("services.sync.prefs.sync.ui.osk.enabled", true);
pref("services.sync.prefs.dangerously_allow_arbitrary", false);
pref("services.sync.syncedTabs.showRemoteIcons", true);
pref("services.sync.syncedTabs.syncDelayAfterTabChange", 0);
pref("browser.menu.showCharacterEncoding", "chrome://browser/locale/browser.properties");
pref("prompts.defaultModalType", 3);
pref("browser.topsites.useRemoteSetting", true);
pref("browser.topsites.contile.enabled", true);
pref("browser.topsites.contile.endpoint", "https://contile.services.mozilla.com/v1/tiles");
pref("browser.partnerlink.attributionURL", "https://topsites.services.mozilla.com/cid/");
pref("browser.partnerlink.campaign.topsites", "amzn_2020_a1");
pref("prompts.tabChromePromptSubDialog", true);
pref("prompts.contentPromptSubDialog", true);
pref("prompts.windowPromptSubDialog", true);
pref("browser.newtab.preload", true);
pref("browser.newtabpage.activity-stream.newNewtabExperience.colors", "#0090ED,#FF4F5F,#2AC3A2,#FF7139,#A172FF,#FFA437,#FF2A8A");
//@line 1507 "$SRCDIR/browser/app/profile/firefox.js"
pref("browser.newtabpage.activity-stream.fxaccounts.endpoint", "https://accounts.firefox.com/");
pref("browser.newtabpage.activity-stream.improvesearch.topSiteSearchShortcuts", true);
pref("browser.newtabpage.activity-stream.asrouter.providers.cfr", "{\"id\":\"cfr\",\"enabled\":true,\"type\":\"remote-settings\",\"bucket\":\"cfr\",\"updateCycleInMs\":3600000}");
pref("browser.newtabpage.activity-stream.asrouter.providers.whats-new-panel", "{\"id\":\"whats-new-panel\",\"enabled\":true,\"type\":\"remote-settings\",\"bucket\":\"whats-new-panel\",\"updateCycleInMs\":3600000}");
pref("browser.newtabpage.activity-stream.asrouter.providers.message-groups", "{\"id\":\"message-groups\",\"enabled\":true,\"type\":\"remote-settings\",\"bucket\":\"message-groups\",\"updateCycleInMs\":3600000}");
pref("browser.newtabpage.activity-stream.asrouter.providers.snippets", "{\"id\":\"snippets\",\"enabled\":false,\"type\":\"remote\",\"url\":\"https://snippets.cdn.mozilla.net/%STARTPAGE_VERSION%/%NAME%/%VERSION%/%APPBUILDID%/%BUILD_TARGET%/%LOCALE%/%CHANNEL%/%OS_VERSION%/%DISTRIBUTION%/%DISTRIBUTION_VERSION%/\",\"updateCycleInMs\":14400000}");
pref("browser.newtabpage.activity-stream.asrouter.providers.messaging-experiments", "{\"id\":\"messaging-experiments\",\"enabled\":true,\"type\":\"remote-experiments\",\"messageGroups\":[\"cfr\",\"aboutwelcome\",\"infobar\",\"spotlight\",\"moments-page\",\"pbNewtab\"],\"updateCycleInMs\":3600000}");
pref("browser.newtabpage.activity-stream.asrouter.userprefs.cfr.addons", true);
pref("browser.newtabpage.activity-stream.asrouter.userprefs.cfr.features", true);
pref("browser.newtabpage.activity-stream.asrouter.useRemoteL10n", true);
pref("browser.newtabpage.activity-stream.discoverystream.enabled", true);
pref("browser.newtabpage.activity-stream.discoverystream.hardcoded-basic-layout", false);
pref("browser.newtabpage.activity-stream.discoverystream.compactLayout.enabled", false);
pref("browser.newtabpage.activity-stream.discoverystream.hybridLayout.enabled", false);
pref("browser.newtabpage.activity-stream.discoverystream.hideCardBackground.enabled", false);
pref("browser.newtabpage.activity-stream.discoverystream.fourCardLayout.enabled", false);
pref("browser.newtabpage.activity-stream.discoverystream.loadMore.enabled", false);
pref("browser.newtabpage.activity-stream.discoverystream.lastCardMessage.enabled", false);
pref("browser.newtabpage.activity-stream.discoverystream.newFooterSection.enabled", false);
pref("browser.newtabpage.activity-stream.discoverystream.saveToPocketCard.enabled", false);
pref("browser.newtabpage.activity-stream.discoverystream.hideDescriptions.enabled", false);
pref("browser.newtabpage.activity-stream.discoverystream.compactGrid.enabled", false);
pref("browser.newtabpage.activity-stream.discoverystream.compactImages.enabled", false);
pref("browser.newtabpage.activity-stream.discoverystream.imageGradient.enabled", false);
pref("browser.newtabpage.activity-stream.discoverystream.titleLines", 3);
pref("browser.newtabpage.activity-stream.discoverystream.descLines", 3);
pref("browser.newtabpage.activity-stream.discoverystream.readTime.enabled", true);
pref("browser.newtabpage.activity-stream.discoverystream.newSponsoredLabel.enabled", false);
pref("browser.newtabpage.activity-stream.discoverystream.essentialReadsHeader.enabled", false);
pref("browser.newtabpage.activity-stream.discoverystream.editorsPicksHeader.enabled", false);
pref("browser.newtabpage.activity-stream.discoverystream.spoc-positions", "1,5,7,11,18,20");
pref("browser.newtabpage.activity-stream.discoverystream.widget-positions", "");
pref("browser.newtabpage.activity-stream.discoverystream.spocs-endpoint", "");
pref("browser.newtabpage.activity-stream.discoverystream.spocs-endpoint-query", "");
pref("browser.newtabpage.activity-stream.discoverystream.sponsored-collections.enabled", false);
pref("browser.newtabpage.activity-stream.discoverystream.sendToPocket.enabled", false);
pref("browser.newtabpage.activity-stream.discoverystream.region-stories-block", "FR");
//@line 1569 "$SRCDIR/browser/app/profile/firefox.js"
  pref("browser.newtabpage.activity-stream.discoverystream.locale-list-config", "");
//@line 1571 "$SRCDIR/browser/app/profile/firefox.js"
pref("browser.newtabpage.activity-stream.discoverystream.region-stories-config", "US,DE,CA,GB,IE,CH,AT,BE,IN");
pref("browser.newtabpage.activity-stream.discoverystream.region-spocs-config", "US,CA,DE,GB");
pref("browser.newtabpage.activity-stream.discoverystream.region-basic-config", "");
pref("browser.newtabpage.activity-stream.discoverystream.isCollectionDismissible", true);
pref("browser.newtabpage.activity-stream.discoverystream.personalization.enabled", true);
pref("browser.newtabpage.activity-stream.discoverystream.personalization.modelKeys", "nb_model_arts_and_entertainment, nb_model_autos_and_vehicles, nb_model_beauty_and_fitness, nb_model_blogging_resources_and_services, nb_model_books_and_literature, nb_model_business_and_industrial, nb_model_computers_and_electronics, nb_model_finance, nb_model_food_and_drink, nb_model_games, nb_model_health, nb_model_hobbies_and_leisure, nb_model_home_and_garden, nb_model_internet_and_telecom, nb_model_jobs_and_education, nb_model_law_and_government, nb_model_online_communities, nb_model_people_and_society, nb_model_pets_and_animals, nb_model_real_estate, nb_model_reference, nb_model_science, nb_model_shopping, nb_model_sports, nb_model_travel");
pref("browser.newtabpage.activity-stream.discoverystream.recs.personalized", false);
pref("browser.newtabpage.activity-stream.discoverystream.spocs.personalized", true);
pref("browser.newtabpage.activity-stream.feeds.section.topstories", true);
pref("browser.newtabpage.activity-stream.improvesearch.handoffToAwesomebar", true);
pref("browser.newtabpage.activity-stream.logowordmark.alwaysVisible", true);
pref("browser.newtabpage.activity-stream.hideTopSitesWithSearchParam", "mfadid=adm");
pref("trailhead.firstrun.newtab.triplets", "");
pref("browser.aboutwelcome.enabled", true);
pref("browser.aboutwelcome.screens", "");
pref("browser.aboutwelcome.skipFocus", true);
pref("browser.messaging-system.whatsNewPanel.enabled", true);
pref("messaging-system.log", "warn");
pref("messaging-system.rsexperimentloader.enabled", true);
pref("messaging-system.rsexperimentloader.collection_id", "nimbus-desktop-experiments");
pref("nimbus.debug", false);
pref("full-screen-api.enabled", true);
pref("toolkit.startup.max_resumed_crashes", 3);
//@line 1636 "$SRCDIR/browser/app/profile/firefox.js"
pref("pdfjs.firstRun", true);
pref("pdfjs.previousHandler.preferredAction", 0);
pref("pdfjs.previousHandler.alwaysAskBeforeHandling", false);
pref("pdfjs.handleOctetStream", true);
pref("sidebar.position_start", true);
pref("security.identitypopup.recordEventTelemetry", true);
pref("security.protectionspopup.recordEventTelemetry", true);
pref("security.app_menu.recordEventTelemetry", true);
pref("security.mixed_content.block_active_content", true);
pref("security.insecure_field_warning.contextual.enabled", true);
pref("security.insecure_connection_icon.enabled", true);
pref("security.insecure_connection_icon.pbmode.enabled", true);
pref("security.insecure_connection_text.enabled", false);
pref("security.insecure_connection_text.pbmode.enabled", false);
pref("security.cert_pinning.enforcement_level", 1);
pref("dom.debug.propagate_gesture_events_through_content", false);
pref("browser.uiCustomization.debug", false);
pref("browser.uiCustomization.state", "");
pref("identity.fxaccounts.enabled", true);
pref("identity.fxaccounts.remote.root", "https://accounts.firefox.com/");
pref("identity.fxaccounts.contextParam", "fx_desktop_v3");
pref("identity.fxaccounts.remote.profile.uri", "https://profile.accounts.firefox.com/v1");
pref("identity.fxaccounts.remote.oauth.uri", "https://oauth.accounts.firefox.com/v1");
pref("identity.fxaccounts.pairing.enabled", true);
pref("identity.fxaccounts.remote.pairing.uri", "wss://channelserver.services.mozilla.com");
pref("identity.sync.tokenserver.uri", "https://token.services.mozilla.com/1.0/sync/1.5");
pref("identity.fxaccounts.autoconfig.uri", "");
pref("identity.sendtabpromo.url", "https://support.mozilla.org/1/firefox/%VERSION%/%OS%/%LOCALE%/send-tab");
pref("identity.mobilepromo.android", "https://www.mozilla.org/firefox/android/?utm_source=firefox-browser&utm_medium=firefox-browser&utm_campaign=");
pref("identity.mobilepromo.ios", "https://www.mozilla.org/firefox/ios/?utm_source=firefox-browser&utm_medium=firefox-browser&utm_campaign=");
//@line 1728 "$SRCDIR/browser/app/profile/firefox.js"
  pref("identity.fxaccounts.migrateToDevEdition", false);
//@line 1730 "$SRCDIR/browser/app/profile/firefox.js"
pref("identity.fxaccounts.commands.enabled", true);
pref("identity.fxaccounts.commands.missed.fetch_interval", 86400);
pref("media.gmp.trial-create.enabled", true);
//@line 1753 "$SRCDIR/browser/app/profile/firefox.js"
pref("media.gmp-gmpopenh264.visible", true);
pref("media.gmp-gmpopenh264.enabled", true);
pref("media.autoplay.default", 1); // 0=Allowed, 1=Blocked, 5=All Blocked
//@line 1764 "$SRCDIR/browser/app/profile/firefox.js"
  pref("media.autoplay.block-webaudio", false);
//@line 1766 "$SRCDIR/browser/app/profile/firefox.js"
pref("media.videocontrols.picture-in-picture.enabled", true);
pref("media.videocontrols.picture-in-picture.audio-toggle.enabled", true);
pref("media.videocontrols.picture-in-picture.video-toggle.enabled", true);
pref("media.videocontrols.picture-in-picture.video-toggle.visibility-threshold", "1.0");
pref("media.videocontrols.picture-in-picture.keyboard-controls.enabled", true);
pref("browser.translation.detectLanguage", false);
pref("browser.translation.neverForLanguages", "");
pref("browser.translation.ui.show", false);
pref("browser.translation.engine", "Google");
pref("toolkit.telemetry.archive.enabled", true);
pref("toolkit.telemetry.shutdownPingSender.enabled", true);
pref("toolkit.telemetry.shutdownPingSender.enabledFirstSession", false);
pref("toolkit.telemetry.firstShutdownPing.enabled", true);
pref("toolkit.telemetry.newProfilePing.enabled", true);
pref("toolkit.telemetry.updatePing.enabled", true);
pref("toolkit.telemetry.bhrPing.enabled", true);
pref("browser.ping-centre.telemetry", true);
pref("browser.ping-centre.log", false);
pref("media.gmp-provider.enabled", true);
pref("network.cookie.cookieBehavior", 5 /* BEHAVIOR_REJECT_TRACKER_AND_PARTITION_FOREIGN */);
pref("privacy.restrict3rdpartystorage.rollout.preferences.TCPToggleInStandard", false);
pref("privacy.restrict3rdpartystorage.rollout.preferences.learnMoreURLSuffix", "total-cookie-protection");
pref("network.cookie.cookieBehavior.pbmode", 5 /* BEHAVIOR_REJECT_TRACKER_AND_PARTITION_FOREIGN */);
pref("privacy.trackingprotection.fingerprinting.enabled", true);
pref("privacy.trackingprotection.cryptomining.enabled", true);
pref("browser.contentblocking.database.enabled", true);
pref("dom.storage_access.enabled", true);
//@line 1830 "$SRCDIR/browser/app/profile/firefox.js"
pref("browser.contentblocking.cryptomining.preferences.ui.enabled", true);
pref("browser.contentblocking.fingerprinting.preferences.ui.enabled", true);
pref("browser.contentblocking.reject-and-isolate-cookies.preferences.ui.enabled", true);
pref("browser.contentblocking.features.strict", "tp,tpPrivate,cm,fp,stp,lvl2,rp,rpTop,ocsp,qps");
pref("browser.contentblocking.customBlockList.preferences.ui.enabled", false);
pref("browser.contentblocking.reportBreakage.url", "https://tracking-protection-issues.herokuapp.com/new");
pref("browser.contentblocking.report.lockwise.enabled", true);
pref("browser.contentblocking.report.monitor.enabled", true);
pref("browser.contentblocking.report.proxy.enabled", false);
pref("browser.contentblocking.report.show_mobile_app", true);
pref("browser.send_to_device_locales", "de,en-GB,en-US,es-AR,es-CL,es-ES,es-MX,fr,id,pl,pt-BR,ru,zh-TW");
pref("browser.vpn_promo.disallowed_regions", "ae,by,cn,cu,iq,ir,kp,om,ru,sd,sy,tm,tr,ua");
pref("browser.vpn_promo.enabled", true);
pref("browser.contentblocking.report.vpn_regions", "as,at,be,ca,ch,de,es,fi,fr,gb,gg,ie,im,io,it,je,mp,my,nl,nz,pr,se,sg,uk,um,us,vg,vi"
);
pref("browser.promo.focus.disallowed_regions", "cn");
pref("browser.promo.focus.enabled", true);
pref("browser.contentblocking.report.vpn_platforms", "win,mac,linux");
pref("browser.contentblocking.report.hide_vpn_banner", false);
pref("browser.contentblocking.report.vpn_sub_id", "sub_HrfCZF7VPHzZkA");
pref("browser.contentblocking.report.monitor.url", "https://monitor.firefox.com/?entrypoint=protection_report_monitor&utm_source=about-protections");
pref("browser.contentblocking.report.monitor.how_it_works.url", "https://monitor.firefox.com/about");
pref("browser.contentblocking.report.monitor.sign_in_url", "https://monitor.firefox.com/oauth/init?entrypoint=protection_report_monitor&utm_source=about-protections&email=");
pref("browser.contentblocking.report.monitor.preferences_url", "https://monitor.firefox.com/user/preferences");
pref("browser.contentblocking.report.monitor.home_page_url", "https://monitor.firefox.com/user/dashboard");
pref("browser.contentblocking.report.manage_devices.url", "https://accounts.firefox.com/settings/clients");
pref("browser.contentblocking.report.endpoint_url", "https://monitor.firefox.com/user/breach-stats?includeResolved=true");
pref("browser.contentblocking.report.proxy_extension.url", "https://fpn.firefox.com/browser?utm_source=firefox-desktop&utm_medium=referral&utm_campaign=about-protections&utm_content=about-protections");
pref("browser.contentblocking.report.mobile-ios.url", "https://apps.apple.com/app/firefox-private-safe-browser/id989804926");
pref("browser.contentblocking.report.mobile-android.url", "https://play.google.com/store/apps/details?id=org.mozilla.firefox&referrer=utm_source%3Dprotection_report%26utm_content%3Dmobile_promotion");
pref("browser.contentblocking.report.vpn.url", "https://vpn.mozilla.org/?utm_source=firefox-browser&utm_medium=firefox-browser&utm_campaign=about-protections-card");
pref("browser.contentblocking.report.vpn-promo.url", "https://vpn.mozilla.org/?utm_source=firefox-browser&utm_medium=firefox-browser&utm_campaign=about-protections-top-promo");
pref("browser.contentblocking.report.vpn-android.url", "https://play.google.com/store/apps/details?id=org.mozilla.firefox.vpn&referrer=utm_source%3Dfirefox-browser%26utm_medium%3Dfirefox-browser%26utm_campaign%3Dabout-protections-mobile-vpn%26anid%3D--");
pref("browser.contentblocking.report.vpn-ios.url", "https://apps.apple.com/us/app/firefox-private-network-vpn/id1489407738");
pref("browser.contentblocking.report.lockwise.how_it_works.url", "https://support.mozilla.org/1/firefox/%VERSION%/%OS%/%LOCALE%/password-manager-report");
pref("browser.contentblocking.report.social.url", "https://support.mozilla.org/1/firefox/%VERSION%/%OS%/%LOCALE%/social-media-tracking-report");
pref("browser.contentblocking.report.cookie.url", "https://support.mozilla.org/1/firefox/%VERSION%/%OS%/%LOCALE%/cross-site-tracking-report");
pref("browser.contentblocking.report.tracker.url", "https://support.mozilla.org/1/firefox/%VERSION%/%OS%/%LOCALE%/tracking-content-report");
pref("browser.contentblocking.report.fingerprinter.url", "https://support.mozilla.org/1/firefox/%VERSION%/%OS%/%LOCALE%/fingerprinters-report");
pref("browser.contentblocking.report.cryptominer.url", "https://support.mozilla.org/1/firefox/%VERSION%/%OS%/%LOCALE%/cryptominers-report");
pref("browser.contentblocking.cfr-milestone.enabled", true);
pref("browser.contentblocking.cfr-milestone.milestone-achieved", 0);
pref("browser.contentblocking.cfr-milestone.milestones", "[1000, 5000, 10000, 25000, 50000, 100000, 250000, 314159, 500000, 750000, 1000000, 1250000, 1500000, 1750000, 2000000, 2250000, 2500000, 8675309]");
//@line 1965 "$SRCDIR/browser/app/profile/firefox.js"
pref("privacy.usercontext.about_newtab_segregation.enabled", true);
//@line 1973 "$SRCDIR/browser/app/profile/firefox.js"
  pref("privacy.userContext.enabled", false);
  pref("privacy.userContext.ui.enabled", false);
//@line 1976 "$SRCDIR/browser/app/profile/firefox.js"
pref("privacy.userContext.extension", "");
pref("privacy.userContext.newTabContainerOnLeftClick.enabled", false);
//@line 1990 "$SRCDIR/browser/app/profile/firefox.js"
pref("privacy.webrtc.allowSilencingNotifications", false);
pref("privacy.webrtc.legacyGlobalIndicator", true);
//@line 1993 "$SRCDIR/browser/app/profile/firefox.js"
pref("privacy.webrtc.globalMuteToggles", false);
pref("privacy.webrtc.sharedTabWarning", false);
pref("privacy.webrtc.deviceGracePeriodTimeoutMs", 3600000);
pref("browser.tabs.remote.autostart", true);
pref("browser.tabs.remote.desktopbehavior", true);
pref("media.peerconnection.mtransport_process", true);
pref("browser.tabs.remote.warmup.enabled", true);
pref("browser.tabs.remote.tabCacheSize", 0);
pref("browser.tabs.remote.warmup.maxTabs", 3);
pref("browser.tabs.remote.warmup.unloadDelayMs", 2000);
pref("browser.tabs.crashReporting.sendReport", true);
pref("browser.tabs.crashReporting.includeURL", false);
pref("extensions.experiments.enabled", false);
//@line 2039 "$SRCDIR/browser/app/profile/firefox.js"
pref("reader.parse-node-limit", 0);
pref("reader.errors.includeURLs", true);
pref("reader.pocket.ctaVersion", "");
pref("view_source.tab", true);
pref("dom.serviceWorkers.enabled", true);
pref("dom.push.enabled", true);
pref("toolkit.pageThumbs.minWidth", 280);
pref("toolkit.pageThumbs.minHeight", 190);
pref("media.webspeech.synth.enabled", true);
pref("browser.esedbreader.loglevel", "Error");
pref("browser.laterrun.enabled", false);
//@line 2073 "$SRCDIR/browser/app/profile/firefox.js"
pref("dom.ipc.processPrelaunch.enabled", true);
//@line 2075 "$SRCDIR/browser/app/profile/firefox.js"
pref("browser.migrate.chrome.history.limit", 2000);
pref("browser.migrate.chrome.history.maxAgeInDays", 180);
pref("browser.migrate.showBookmarksToolbarAfterMigration", true);
pref("extensions.pocket.api", "api.getpocket.com");
pref("extensions.pocket.enabled", true);
pref("extensions.pocket.oAuthConsumerKey", "40249-e88c401e1b1f2242d9e441c4");
pref("extensions.pocket.site", "getpocket.com");
pref("extensions.pocket.onSaveRecs", true);
pref("extensions.pocket.onSaveRecs.locales", "en-US,en-GB,en-CA");
pref("extensions.pocket.showHome", true);
pref("extensions.pocket.loggedOutVariant", "control");
pref("extensions.pocket.refresh.layout.enabled", true);
pref("extensions.pocket.refresh.emailButton.enabled", false);
pref("extensions.pocket.refresh.hideRecentSaves.enabled", false);
pref("signon.management.page.fileImport.enabled", false);
//@line 2108 "$SRCDIR/browser/app/profile/firefox.js"
pref("signon.management.page.os-auth.enabled", false);
//@line 2110 "$SRCDIR/browser/app/profile/firefox.js"
pref("signon.management.page.breach-alerts.enabled", true);
pref("signon.management.page.vulnerable-passwords.enabled", true);
pref("signon.management.page.sort", "name");
pref("signon.management.page.breachAlertUrl",
     "https://monitor.firefox.com/breach-details/");
pref("signon.passwordEditCapture.enabled", true);
pref("signon.relatedRealms.enabled", false);
pref("signon.showAutoCompleteFooter", true);
pref("signon.showAutoCompleteImport", "import");
pref("signon.suggestImportCount", 3);
pref("print.use_simplify_page", true);
pref("webchannel.allowObject.urlWhitelist", "https://content.cdn.mozilla.net https://support.mozilla.org https://install.mozilla.org");
//@line 2138 "$SRCDIR/browser/app/profile/firefox.js"
  pref("browser.crashReports.unsubmittedCheck.enabled", false);
//@line 2140 "$SRCDIR/browser/app/profile/firefox.js"
pref("browser.crashReports.unsubmittedCheck.chancesUntilSuppress", 4);
pref("browser.crashReports.unsubmittedCheck.autoSubmit2", false);
pref("services.sync.engine.creditcards.available", true);
pref("extensions.formautofill.reauth.enabled", false);
pref("browser.sessionstore.restore_tabs_lazily", true);
pref("browser.suppress_first_window_animation", true);
pref("extensions.screenshots.disabled", false);
pref("screenshots.browser.component.enabled", false);
pref("doh-rollout.clearModeOnShutdown", false);
pref("browser.chrome.errorReporter.infoURL",
     "https://support.mozilla.org/1/firefox/%VERSION%/%OS%/%LOCALE%/nightly-error-collection");
pref("app.normandy.api_url", "https://normandy.cdn.mozilla.net/api/v1");
pref("app.normandy.dev_mode", false);
pref("app.normandy.enabled", true);
pref("app.normandy.first_run", true);
pref("app.normandy.logging.level", 50); // Warn
pref("app.normandy.run_interval_seconds", 21600); // 6 hours
pref("app.normandy.shieldLearnMoreUrl", "https://support.mozilla.org/1/firefox/%VERSION%/%OS%/%LOCALE%/shield");
pref("app.normandy.last_seen_buildid", "");
pref("app.normandy.onsync_skew_sec", 600);
//@line 2183 "$SRCDIR/browser/app/profile/firefox.js"
  pref("app.shield.optoutstudies.enabled", true);
//@line 2187 "$SRCDIR/browser/app/profile/firefox.js"
//@line 2201 "$SRCDIR/browser/app/profile/firefox.js"
  pref("intl.multilingual.enabled", true);
  pref("intl.multilingual.downloadEnabled", true);
  pref("intl.multilingual.liveReload", true);
  pref("intl.multilingual.liveReloadBidirectional", false);
  pref("intl.multilingual.aboutWelcome.languageMismatchEnabled", true);
//@line 2213 "$SRCDIR/browser/app/profile/firefox.js"
pref("fission.frontend.simulate-events", false);
pref("fission.frontend.simulate-messages", false);
pref("toolkit.coverage.enabled", false);
pref("toolkit.coverage.endpoint.base", "https://coverage.mozilla.org");
pref("browser.discovery.enabled", true);
pref("browser.discovery.containers.enabled", true);
pref("browser.discovery.sites", "addons.mozilla.org");
pref("browser.engagement.recent_visited_origins.expiry", 86400); // 24 * 60 * 60 (24 hours in seconds)
pref("browser.engagement.downloads-button.has-used", false);
pref("browser.engagement.fxa-toolbar-menu-button.has-used", false);
pref("browser.engagement.home-button.has-used", false);
pref("browser.engagement.sidebar-button.has-used", false);
pref("browser.engagement.library-button.has-used", false);
pref("browser.engagement.ctrlTab.has-used", false);
pref("browser.aboutConfig.showWarning", true);
pref("browser.toolbars.keyboard_navigation", true);
pref("browser.toolbars.bookmarks.visibility", "newtab");
pref("browser.toolbars.bookmarks.showOtherBookmarks", true);
pref("identity.fxaccounts.toolbar.enabled", true);
pref("identity.fxaccounts.toolbar.accessed", false);
pref("identity.fxaccounts.service.monitorLoginUrl", "https://monitor.firefox.com/");
pref("corroborator.enabled", true);
pref("devtools.toolbox.footer.height", 250);
pref("devtools.toolbox.sidebar.width", 500);
pref("devtools.toolbox.host", "bottom");
pref("devtools.toolbox.previousHost", "right");
pref("devtools.toolbox.selectedTool", "inspector");
pref("devtools.toolbox.sideEnabled", true);
pref("devtools.toolbox.zoomValue", "1");
pref("devtools.toolbox.splitconsoleEnabled", false);
pref("devtools.toolbox.splitconsoleHeight", 100);
pref("devtools.toolbox.tabsOrder", "");
//@line 2286 "$SRCDIR/browser/app/profile/firefox.js"
pref("devtools.browsertoolbox.fission", false);
//@line 2288 "$SRCDIR/browser/app/profile/firefox.js"
pref("devtools.target-switching.server.enabled", true);
pref("devtools.every-frame-target.enabled", true);
pref("devtools.popups.debug", false);
pref("devtools.command-button-pick.enabled", true);
pref("devtools.command-button-frames.enabled", true);
pref("devtools.command-button-splitconsole.enabled", true);
pref("devtools.command-button-responsive.enabled", true);
pref("devtools.command-button-screenshot.enabled", false);
pref("devtools.command-button-rulers.enabled", false);
pref("devtools.command-button-measure.enabled", false);
pref("devtools.command-button-noautohide.enabled", false);
pref("devtools.command-button-errorcount.enabled", true);
//@line 2313 "$SRCDIR/browser/app/profile/firefox.js"
pref("devtools.inspector.enabled", true);
pref("devtools.inspector.selectedSidebar", "layoutview");
pref("devtools.inspector.activeSidebar", "layoutview");
pref("devtools.inspector.remote", false);
pref("devtools.inspector.three-pane-enabled", true);
pref("devtools.inspector.chrome.three-pane-enabled", false);
pref("devtools.inspector.show_pseudo_elements", false);
pref("devtools.inspector.imagePreviewTooltipSize", 300);
pref("devtools.inspector.showUserAgentStyles", false);
pref("devtools.inspector.showAllAnonymousContent", false);
pref("devtools.inspector.ruleview.inline-compatibility-warning.enabled", false);
pref("devtools.inspector.compatibility.enabled", true);
pref("devtools.overflow.debugging.enabled", true);
pref("devtools.inspector.draggable_properties", true);
pref("devtools.gridinspector.gridOutlineMaxColumns", 50);
pref("devtools.gridinspector.gridOutlineMaxRows", 50);
pref("devtools.gridinspector.showGridAreas", false);
pref("devtools.gridinspector.showGridLineNumbers", false);
pref("devtools.gridinspector.showInfiniteLines", false);
pref("devtools.gridinspector.maxHighlighters", 3);
pref("devtools.layout.boxmodel.opened", true);
pref("devtools.layout.flexbox.opened", true);
pref("devtools.layout.flex-container.opened", true);
pref("devtools.layout.flex-item.opened", true);
pref("devtools.layout.grid.opened", true);
//@line 2368 "$SRCDIR/browser/app/profile/firefox.js"
  pref("devtools.layout.boxmodel.highlightProperty", false);
//@line 2370 "$SRCDIR/browser/app/profile/firefox.js"
pref("devtools.eyedropper.zoom", 6);
pref("devtools.markup.collapseAttributes", true);
pref("devtools.markup.collapseAttributeLength", 120);
pref("devtools.markup.beautifyOnCopy", false);
pref("devtools.markup.mutationBreakpoints.enabled", true);
pref("devtools.defaultColorUnit", "authored");
pref("devtools.memory.enabled", true);
pref("devtools.memory.custom-census-displays", "{}");
pref("devtools.memory.custom-label-displays", "{}");
pref("devtools.memory.custom-tree-map-displays", "{}");
pref("devtools.memory.max-individuals", 1000);
pref("devtools.memory.max-retaining-paths", 10);
pref("devtools.performance.enabled", true);
pref("devtools.cache.disabled", false);
pref("devtools.serviceWorkers.testing.enabled", false);
pref("devtools.netmonitor.enabled", true);
pref("devtools.netmonitor.features.search", true);
pref("devtools.netmonitor.features.requestBlocking", true);
pref("devtools.application.enabled", true);
pref("devtools.custom-formatters", false);
pref("devtools.custom-formatters.enabled", false);
pref("devtools.netmonitor.panes-network-details-width", 550);
pref("devtools.netmonitor.panes-network-details-height", 450);
pref("devtools.netmonitor.panes-search-width", 550);
pref("devtools.netmonitor.panes-search-height", 450);
pref("devtools.netmonitor.filters", "[\"all\"]");
pref("devtools.netmonitor.visibleColumns",
    "[\"status\",\"method\",\"domain\",\"file\",\"initiator\",\"type\",\"transferred\",\"contentSize\",\"waterfall\"]"
);
pref("devtools.netmonitor.columnsData",
  '[{"name":"status","minWidth":30,"width":5}, {"name":"method","minWidth":30,"width":5}, {"name":"domain","minWidth":30,"width":10}, {"name":"file","minWidth":30,"width":25}, {"name":"url","minWidth":30,"width":25},{"name":"initiator","minWidth":30,"width":10},{"name":"type","minWidth":30,"width":5},{"name":"transferred","minWidth":30,"width":10},{"name":"contentSize","minWidth":30,"width":5},{"name":"waterfall","minWidth":150,"width":15}]');
pref("devtools.netmonitor.msg.payload-preview-height", 128);
pref("devtools.netmonitor.msg.visibleColumns",
  '["data", "time"]'
);
pref("devtools.netmonitor.msg.displayed-messages.limit", 500);
pref("devtools.netmonitor.response.ui.limit", 10240);
pref("devtools.netmonitor.saveRequestAndResponseBodies", true);
pref("devtools.netmonitor.har.defaultLogDir", "");
pref("devtools.netmonitor.har.defaultFileName", "%hostname_Archive [%date]");
pref("devtools.netmonitor.har.jsonp", false);
pref("devtools.netmonitor.har.jsonpCallback", "");
pref("devtools.netmonitor.har.includeResponseBodies", true);
pref("devtools.netmonitor.har.compress", false);
pref("devtools.netmonitor.har.forceExport", false);
pref("devtools.netmonitor.har.pageLoadedTimeout", 1500);
pref("devtools.netmonitor.har.enableAutoExportToFile", false);
pref("devtools.netmonitor.features.webSockets", true);
pref("devtools.netmonitor.audits.slow", 500);
pref("devtools.netmonitor.features.serverSentEvents", true);
  pref("devtools.netmonitor.features.newEditAndResend", true);
pref("devtools.netmonitor.customRequest", '{}');
pref("devtools.storage.enabled", true);
pref("devtools.styleeditor.enabled", true);
pref("devtools.styleeditor.autocompletion-enabled", true);
pref("devtools.styleeditor.showMediaSidebar", true);
pref("devtools.styleeditor.mediaSidebarWidth", 238);
pref("devtools.styleeditor.navSidebarWidth", 245);
pref("devtools.styleeditor.transitions", true);
pref("devtools.screenshot.clipboard.enabled", false);
pref("devtools.screenshot.audio.enabled", true);
pref("devtools.dom.enabled", false);
pref("devtools.accessibility.enabled", true);
pref("devtools.webconsole.filter.error", true);
pref("devtools.webconsole.filter.warn", true);
pref("devtools.webconsole.filter.info", true);
pref("devtools.webconsole.filter.log", true);
pref("devtools.webconsole.filter.debug", true);
pref("devtools.webconsole.filter.css", false);
pref("devtools.webconsole.filter.net", false);
pref("devtools.webconsole.filter.netxhr", false);
pref("devtools.webconsole.input.autocomplete",true);
pref("devtools.webconsole.input.context", true);
pref("devtools.webconsole.input.eagerEvaluation", true);
pref("devtools.browserconsole.filter.error", true);
pref("devtools.browserconsole.filter.warn", true);
pref("devtools.browserconsole.filter.info", true);
pref("devtools.browserconsole.filter.log", true);
pref("devtools.browserconsole.filter.debug", true);
pref("devtools.browserconsole.filter.css", false);
pref("devtools.browserconsole.filter.net", false);
pref("devtools.browserconsole.filter.netxhr", false);
pref("devtools.webconsole.inputHistoryCount", 300);
pref("devtools.webconsole.persistlog", false);
pref("devtools.netmonitor.persistlog", false);
pref("devtools.webconsole.timestampMessages", false);
//@line 2537 "$SRCDIR/browser/app/profile/firefox.js"
  pref("devtools.webconsole.sidebarToggle", false);
//@line 2539 "$SRCDIR/browser/app/profile/firefox.js"
pref("devtools.webconsole.input.editor", false);
pref("devtools.browserconsole.input.editor", false);
pref("devtools.webconsole.input.editorWidth", 0);
pref("devtools.browserconsole.input.editorWidth", 0);
pref("devtools.webconsole.input.editorOnboarding", true);
pref("devtools.webconsole.groupWarningMessages", true);
pref("devtools.browserconsole.contentMessages", false);
pref("devtools.source-map.client-service.enabled", true);
pref("devtools.hud.loglimit", 10000);
pref("devtools.editor.tabsize", 2);
pref("devtools.editor.expandtab", true);
pref("devtools.editor.keymap", "default");
pref("devtools.editor.autoclosebrackets", true);
pref("devtools.editor.detectindentation", true);
pref("devtools.editor.enableCodeFolding", true);
pref("devtools.editor.autocomplete", true);
pref("devtools.responsive.viewport.angle", 0);
pref("devtools.responsive.viewport.width", 320);
pref("devtools.responsive.viewport.height", 480);
pref("devtools.responsive.viewport.pixelRatio", 0);
pref("devtools.responsive.leftAlignViewport.enabled", false);
pref("devtools.responsive.reloadConditions.touchSimulation", false);
pref("devtools.responsive.reloadConditions.userAgent", false);
pref("devtools.responsive.reloadNotification.enabled", true);
pref("devtools.responsive.touchSimulation.enabled", false);
pref("devtools.responsive.userAgent", "");
//@line 2603 "$SRCDIR/browser/app/profile/firefox.js"
  pref("devtools.responsive.showUserAgentInput", false);
//@line 2605 "$SRCDIR/browser/app/profile/firefox.js"
//@line 2608 "$SRCDIR/browser/app/profile/firefox.js"
  pref("devtools.aboutdebugging.local-tab-debugging", false);
//@line 2612 "$SRCDIR/browser/app/profile/firefox.js"
pref("devtools.aboutdebugging.process-debugging", true);
pref("devtools.aboutdebugging.network-locations", "[]");
pref("devtools.aboutdebugging.collapsibilities.installedExtension", false);
pref("devtools.aboutdebugging.collapsibilities.otherWorker", false);
pref("devtools.aboutdebugging.collapsibilities.serviceWorker", false);
pref("devtools.aboutdebugging.collapsibilities.sharedWorker", false);
pref("devtools.aboutdebugging.collapsibilities.tab", false);
pref("devtools.aboutdebugging.collapsibilities.temporaryExtension", false);
//@line 2628 "$SRCDIR/browser/app/profile/firefox.js"
  pref("devtools.aboutdebugging.showHiddenAddons", false);
//@line 2632 "$SRCDIR/browser/app/profile/firefox.js"
pref("devtools.debugger.features.map-await-expression", true);
pref("devtools.debugger.features.async-captured-stacks", true);
pref("devtools.debugger.features.async-live-stacks", false);
pref("devtools.popup.disable_autohide", false);
pref("first-startup.timeout", 30000);
//@line 2654 "$SRCDIR/browser/app/profile/firefox.js"
//@line 2661 "$SRCDIR/browser/app/profile/firefox.js"
//@line 2666 "$SRCDIR/browser/app/profile/firefox.js"
  pref("browser.menu.showViewImageInfo", false);
//@line 2668 "$SRCDIR/browser/app/profile/firefox.js"
pref("svg.context-properties.content.allowed-domains", "profile.accounts.firefox.com,profile.stage.mozaws.net");
//@line 2680 "$SRCDIR/browser/app/profile/firefox.js"
pref("browser.snapshots.score.Visit", 1);
pref("browser.snapshots.score.CurrentSession", 1);
pref("browser.snapshots.score.IsUserPersisted", 1);
pref("browser.snapshots.score.IsUsedRemoved", -10);
pref("browser.snapshots.source.CommonReferrer", 3);
pref("browser.snapshots.source.Overlapping", 3);
pref("browser.snapshots.source.TimeOfDay", 3);
pref("browser.snapshots.relevancy.timeOfDayIntervalSeconds", 3600);
pref("browser.places.snapshots.expiration.days", 210);
pref("browser.places.snapshots.expiration.userManaged.days", 420);
