from . import TST_CFG_TMPL, TST_CFG, util, new_template, combine, Required, make_template, RequiredError, auto_super, ADDA_CFG_TEMPLATE, COMBO_CFG

if __name__ == '__main__':
    # @auto_super
    class SOME_TEST(TST_CFG_TMPL):
        some_default_value="DEFAULT"
        def __init__(self, *args, **kwargs):
            super(SOME_TEST, self).__init__(*args, **kwargs)

    A=SOME_TEST(
        model_save_subdir="is_default_value_here?"
        # some_default_value=lambda: "Callable"
    )
    # print("VARS:",vars(A.__class__))
    # pout.x(0)
    A().summary()
    class TST_REQ(TST_CFG):
        req_val=Required()
        def __init__(self, *args, **kwargs):
            super(TST_REQ, self).__init__(*args, **kwargs)

    TST_TMPL = TST_CFG(template="included!")
    TST_A = lambda: TST_REQ(
        model_save_subdir="A✓✓",
        req_val="I'm here!",
        values=TST_CFG(
            nm="TST_A",
            b=TST_CFG(
                dotted=util.Dottify(
                    level_two="updated_a✗",
                    from_a="YES✓"
                )
            )
        )
    )
    TST_B = TST_CFG(
        model_save_subdir="B✓✓",
        values=TST_CFG(
            nm="TST_B✓",
            b=TST_CFG(
                dotted=util.Dottify(
                    level_two="updated_b✓",
                    from_b="YES✓"
                )
            )
        )
    )
    TST_MISSING_REQ = make_template(TST_REQ)(
        model_save_subdir="Without required value:"
    )
    TST_ADD_MISSING_REQ = TST_CFG(
        model_save_subdir="added missingvalue:",
        req_val="required value"
    )
    combined_required=combine(TST_MISSING_REQ, TST_ADD_MISSING_REQ, trigger=False)

    TST_REQ_COMBO=new_template(name="REQ_TEST", callback=lambda: combined_required)
    TST_REQ_COMBO().summary()
    try:
        DO_FAIL=new_template(name="SHOULD_FAIL", callback=lambda: TST_MISSING_REQ)
        DO_FAIL().summary()
    except RequiredError as e:
        print("Failed as supposed to (missing req_val): {e}")

    SANITY_TST = new_template(name="TESTING", callback=lambda: combine(TST_TMPL, TST_A, TST_B))
    SANITY_TST().summary()
    pout.x(0)
