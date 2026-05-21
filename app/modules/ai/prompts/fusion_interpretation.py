"""
Prompt for interpreting multimodal fusion results in Turkish.
"""
from app.modules.ai.prompts.templates import PromptTemplate

FUSION_INTERPRETATION_TR = PromptTemplate(
    name="fusion_interpretation_tr",
    template=(
        "Aşağıda HantaProject çok modlu risk değerlendirme sisteminin çıktısı yer alıyor. "
        "Bu sonucu Türkçe, sade ve kısa bir dille kullanıcıya açıkla.\n"
        "\n"
        "--- Risk Değerlendirmesi ---\n"
        "Bütünleşik risk skoru: {final_risk_score:.2f} (0=düşük, 1=yüksek)\n"
        "Risk seviyesi: {risk_level}\n"
        "Genel güven: {fusion_confidence}\n"
        "\n"
        "--- Semptom/Risk Modeli (Birincil Sinyal) ---\n"
        "Tahmin etiketi: {ml_label}\n"
        "Pozitif olasılık: {ml_probability:.2%}\n"
        "Model güveni: {ml_confidence}\n"
        "\n"
        "--- Görüntü Analizi (Destekleyici Kanıt) ---\n"
        "Görüntü kullanıldı mı: {vision_used}\n"
        "Görüntü durumu: {vision_status}\n"
        "Tahmin edilen sınıf: {vision_class}\n"
        "Görüntü güven skoru: {vision_confidence}\n"
        "Reddedilme nedeni: {vision_rejection_reason}\n"
        "\n"
        "--- Belirsizlik Sinyalleri ---\n"
        "{uncertainty_flags}\n"
        "\n"
        "--- Baskın Sinyal ---\n"
        "{dominant_signal}\n"
        "\n"
        "Açıklamanda şunları içer:\n"
        "1. Bütünleşik risk skorunun ne anlama geldiğini ve hangi sinyalin baskın olduğunu sade dille anlat.\n"
        "2. Görüntü analizi kullanıldıysa nasıl katkı sağladığını, kullanılmadıysa nedenini belirt.\n"
        "3. Belirsizlik sinyali varsa bunu basit bir uyarı olarak aktarın.\n"
        "4. Bu sistemin tıbbi teşhis yerine geçmediğini, sonuçların mutlaka bir sağlık uzmanına "
        "danışılarak yorumlanması gerektiğini hatırlat.\n"
        "Yanıt 8 cümleyi geçmesin."
    ),
)
