/* <provider-blocks> — the one implementation of the provider-block selector
 * and the API Keys manager.
 *
 * Markup lives in index.html as <template> elements, so there is exactly one
 * copy of it and the stylesheet/markup guards keep scanning the same file.
 * This module only clones the template for the requested mode, bakes in the
 * per-site props and expands the shared API-key editor.
 *
 * Usage — picker (Engine, Vision, Voice, Hearing, Memory, Thoughts):
 *
 *   <provider-blocks mode="picker"
 *                    category="vision"
 *                    ref="cfg.llm.vision_provider_ref"
 *                    label="Vision provider block"
 *                    hint="…"
 *                    features="key new"
 *                    key-block="visionBlock()"></provider-blocks>
 *
 * Usage — the API Keys page:
 *
 *   <provider-blocks mode="manager"></provider-blocks>
 *
 * Props are read once, when the element upgrades: every call site is static
 * markup. The reactive state — providers, cfg, secretEdits, providerBusy —
 * stays in the single app store, so there is no second copy to keep in sync
 * and x-model binds straight to cfg.* inside the cloned markup.
 *
 * features: "key" renders the inline API-key editor, "new" renders the
 * ＋ new block / save blocks strip. Anything else is refused, so a typo
 * fails at boot instead of silently hiding a control.
 */
(function () {
  "use strict";

  var MODE_TEMPLATE = {
    picker: "tpl-provider-picker",
    manager: "tpl-provider-manager",
  };
  var KEY_TEMPLATE = "tpl-provider-key";
  var KEY_SLOT = "provider-key-slot";
  var FEATURES = ["key", "new"];
  // Placeholders the templates may use. fill() refuses anything else, so a
  // typo in either half of the contract fails loudly instead of printing
  // a literal %%ref%% in the page.
  var PLACEHOLDERS = ["ref", "cat", "catLower", "label", "hint", "auto", "block"];
  var CATEGORY_LABELS = {
    llm: "LLM", vision: "Vision", tts: "TTS",
    stt: "STT", memory: "Memory", thoughts: "Thoughts",
  };

  function escapeText(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function fill(html, values) {
    return html.replace(/%%([A-Za-z]+)%%/g, function (whole, name) {
      if (PLACEHOLDERS.indexOf(name) === -1 ||
          !Object.prototype.hasOwnProperty.call(values, name)) {
        throw new Error("provider-blocks: no value for placeholder " + whole);
      }
      return values[name];
    });
  }

  function templateById(id) {
    var tpl = document.getElementById(id);
    if (!tpl) {
      throw new Error('provider-blocks: missing <template id="' + id + '">');
    }
    return tpl;
  }

  // The templates nest real <template> elements (x-if / x-for); their content
  // lives in a separate DocumentFragment that querySelectorAll does not reach,
  // so slots and feature flags inside them need an explicit walk.
  function eachDeep(root, selector, fn) {
    Array.prototype.forEach.call(root.querySelectorAll(selector), fn);
    Array.prototype.forEach.call(root.querySelectorAll("template"), function (tpl) {
      eachDeep(tpl.content, selector, fn);
    });
  }

  function render(host, mode) {
    var tplId = MODE_TEMPLATE[mode];
    if (!tplId) {
      throw new Error('provider-blocks: unknown mode "' + mode + '"');
    }

    var enabled = (host.getAttribute("features") || "").split(/\s+/).filter(Boolean);
    enabled.forEach(function (feature) {
      if (FEATURES.indexOf(feature) === -1) {
        console.warn("provider-blocks: unknown feature \"" + feature + "\" (known: " + FEATURES.join(", ") + ")");
      }
    });

    var values = {};
    if (mode === "picker") {
      var category = (host.getAttribute("category") || "").trim();
      var ref = (host.getAttribute("ref") || "").trim();
      if (!CATEGORY_LABELS[category]) {
        throw new Error('provider-blocks: unknown category "' + category + '"');
      }
      // Only a plain cfg path is interpolated into Alpine expressions — this
      // is what keeps x-model bound to the real config, not a private copy.
      if (!/^cfg\.[a-z_]+\.[a-z_]+$/.test(ref)) {
        throw new Error('provider-blocks: ref must be a cfg.* path, got "' + ref + '"');
      }
      var categoryLabel = CATEGORY_LABELS[category];
      // visionBlock() is the only resolver the app exposes for "the block this
      // subsystem will actually use". Any other category asking for a key
      // editor has to say which block it means.
      if (enabled.indexOf("key") !== -1 && !host.getAttribute("key-block") && category !== "vision") {
        throw new Error('provider-blocks: features="key" needs key-block="…" outside the vision picker');
      }
      values = {
        ref: ref,
        cat: "'" + category + "'",
        catLower: categoryLabel.toLowerCase(),
        label: escapeText(host.getAttribute("label") || "Provider block"),
        hint: host.getAttribute("hint") || "",
        auto: escapeText(host.getAttribute("auto-label") || "— auto (first " + categoryLabel + " block) —"),
        block: host.getAttribute("key-block") || "visionBlock()",
      };
    }

    var holder = document.createElement("div");
    holder.innerHTML = fill(templateById(tplId).innerHTML, values);

    // A marker may require several features (space-separated = all of them).
    eachDeep(holder, "[data-feature]", function (el) {
      var needed = (el.getAttribute("data-feature") || "").split(/\s+/).filter(Boolean);
      if (needed.some(function (feature) { return enabled.indexOf(feature) === -1; })) {
        el.remove();
      }
    });

    // One API-key editor for the picker and every manager row.
    var keyTemplate = templateById(KEY_TEMPLATE);
    eachDeep(holder, KEY_SLOT, function (slot) {
      var block = slot.getAttribute("block");
      if (!block) {
        throw new Error("provider-blocks: <" + KEY_SLOT + "> without a block expression");
      }
      var keyHolder = document.createElement("div");
      keyHolder.innerHTML = fill(keyTemplate.innerHTML, { block: block });
      slot.replaceWith.apply(slot, Array.prototype.slice.call(keyHolder.childNodes));
    });

    var fragment = document.createDocumentFragment();
    while (holder.firstChild) {
      fragment.appendChild(holder.firstChild);
    }
    host.replaceChildren(fragment);
  }

  class ProviderBlocks extends HTMLElement {
    connectedCallback() {
      if (this._providerBlocksMounted) return;
      this._providerBlocksMounted = true;
      try {
        render(this, this.getAttribute("mode") || "picker");
      } catch (err) {
        // Keep one broken call site from taking the whole page down, but make
        // it findable: the element stays empty and the console names it.
        console.error("provider-blocks:", err, this);
        this.setAttribute("data-error", err.message || String(err));
      }
    }
  }

  if (!customElements.get("provider-blocks")) {
    customElements.define("provider-blocks", ProviderBlocks);
  }
})();
