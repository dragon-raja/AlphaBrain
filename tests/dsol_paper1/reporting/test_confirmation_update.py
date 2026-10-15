from reports.paper1.confirmation_update import number, pp


def test_report_rounds_percentage_ties_consistently():
    assert number(50.625)=='50.63'
    assert number(54.37499999999999)=='54.38'
    assert pp(-.01015625)=='-1.02'
    assert number(-6.25,1,True)=='-6.3'
